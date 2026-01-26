"""
Adaptation Triggers
===================

This module provides trigger classes that determine WHEN grid adaptation should occur
during training. Triggers are lightweight, stateless functions that examine the
AdaptationState and return a boolean indicating whether adaptation should be performed.

Trigger Design Principles
-------------------------
1. **Stateless**: Triggers do not maintain internal state. All necessary information
   comes from the AdaptationState object, which is managed externally.
2. **Lightweight**: Trigger checks are O(1) operations involving only scalar comparisons.
3. **Composable**: Multiple triggers can be combined (future extension).
4. **Extensible**: New triggers can be added by subclassing BaseTrigger.

Available Triggers
------------------
PeriodicTrigger
    Fires at regular intervals (every N epochs).
ScheduledTrigger
    Fires at specific pre-defined epochs.
LossPlateauTrigger
    Fires when the loss stops improving for a specified patience period.

Usage Pattern
-------------
    trigger = LossPlateauTrigger(patience=50, rel_threshold=0.05)
    
    for epoch in range(num_epochs):
        loss = train_step(...)
        state = update_state(state, loss)
        
        if trigger(state):
            model.adapt(...)
            state = reset_after_adaptation(state)

Cooldown Mechanism
------------------
All triggers support an optional cooldown period that prevents rapid successive
adaptations. After an adaptation occurs, the trigger will not fire again until
`cooldown` epochs have passed. This is enforced via the `last_adaptation_epoch`
field in AdaptationState.
"""

from abc import ABC, abstractmethod
from typing import List, Set
from .state import AdaptationState, epochs_since_adaptation


class BaseTrigger(ABC):
    """
    Abstract base class for all adaptation triggers.
    
    Triggers determine when grid adaptation should occur during training.
    They are implemented as callable objects that take an AdaptationState
    and return a boolean.
    
    Subclasses must implement the `should_trigger()` method, which contains
    the trigger-specific logic. The base class `__call__` method handles
    common functionality like cooldown enforcement.
    
    Attributes
    ----------
    cooldown : int
        Minimum number of epochs that must pass after an adaptation before
        this trigger can fire again. Default is 0 (no cooldown).
    
    Notes
    -----
    The cooldown is enforced in the base class `__call__` method, so subclasses
    only need to implement their specific triggering logic in `should_trigger()`.
    
    Examples
    --------
    To create a custom trigger:
    
    >>> class MyTrigger(BaseTrigger):
    ...     def __init__(self, threshold: float, cooldown: int = 0):
    ...         super().__init__(cooldown)
    ...         self.threshold = threshold
    ...     
    ...     def should_trigger(self, state: AdaptationState) -> bool:
    ...         return state.best_loss < self.threshold
    """
    
    def __init__(self, cooldown: int = 0):
        """
        Initialize the base trigger.
        
        Parameters
        ----------
        cooldown : int, optional
            Minimum number of epochs between adaptations. Default is 0.
            If cooldown=10 and an adaptation occurred at epoch 50, the next
            adaptation can occur no earlier than epoch 60.
        
        Raises
        ------
        ValueError
            If cooldown is negative.
        """
        if cooldown < 0:
            raise ValueError(f"cooldown must be non-negative, got {cooldown}")
        self.cooldown = cooldown
    
    def __call__(self, state: AdaptationState) -> bool:
        """
        Check if the trigger should fire.
        
        This method first checks if the cooldown period has elapsed since
        the last adaptation, then delegates to the subclass's should_trigger()
        method for trigger-specific logic.
        
        Parameters
        ----------
        state : AdaptationState
            Current adaptation state containing epoch info and loss tracking.
        
        Returns
        -------
        bool
            True if adaptation should occur, False otherwise.
        
        Notes
        -----
        The cooldown check is: epochs_since_adaptation(state) >= cooldown.
        This means if cooldown=0, the trigger can fire every epoch (if conditions are met).
        If cooldown=10, at least 10 epochs must pass after each adaptation.
        """
        # Check cooldown
        if epochs_since_adaptation(state) < self.cooldown:
            return False
        
        # Delegate to subclass-specific logic
        return self.should_trigger(state)
    
    def should_adapt(self, state: AdaptationState) -> bool:
        """
        Alias for __call__ with a more explicit name.
        
        Parameters
        ----------
        state : AdaptationState
            Current adaptation state.
        
        Returns
        -------
        bool
            True if adaptation should occur, False otherwise.
        """
        return self(state)
    
    @abstractmethod
    def should_trigger(self, state: AdaptationState) -> bool:
        """
        Subclass-specific trigger logic.
        
        This method is called by __call__ after the cooldown check passes.
        Subclasses must implement this method with their specific triggering conditions.
        
        Parameters
        ----------
        state : AdaptationState
            Current adaptation state.
        
        Returns
        -------
        bool
            True if the trigger condition is met, False otherwise.
        """
        pass


class PeriodicTrigger(BaseTrigger):
    """
    Trigger that fires at regular intervals (every N epochs since last adaptation).
    
    This trigger fires when at least `period` epochs have passed since the last
    adaptation (or since epoch 0 if no adaptation has occurred). It's useful for
    scheduled regular adaptations regardless of training dynamics.
    
    Attributes
    ----------
    period : int
        Number of epochs between trigger activations.
    
    Notes
    -----
    The trigger fires when: epochs_since_adaptation >= period
    AND epoch % period == 0 (to maintain regular cadence).
    
    Examples
    --------
    >>> trigger = PeriodicTrigger(period=100)
    >>> 
    >>> # No prior adaptation, fires at epochs 100, 200, ...
    >>> state = AdaptationState(epoch=100, last_adaptation_epoch=-1)
    >>> print(trigger.should_adapt(state))  # True
    >>> 
    >>> state = AdaptationState(epoch=50, last_adaptation_epoch=-1)
    >>> print(trigger.should_adapt(state))  # False
    """
    
    def __init__(self, period: int):
        """
        Initialize the periodic trigger.
        
        Parameters
        ----------
        period : int
            Number of epochs between trigger activations. Must be positive.
            The trigger fires when period epochs have passed since last adaptation.
        
        Raises
        ------
        ValueError
            If period < 1.
        
        Examples
        --------
        >>> trigger = PeriodicTrigger(period=50)  # Fire every 50 epochs
        """
        # Period acts as cooldown
        super().__init__(cooldown=0)
        if period < 1:
            raise ValueError(f"period must be positive, got {period}")
        self.period = period
    
    def should_trigger(self, state: AdaptationState) -> bool:
        """
        Check if enough epochs have passed since last adaptation.
        
        Parameters
        ----------
        state : AdaptationState
            Current adaptation state.
        
        Returns
        -------
        bool
            True if epochs_since_adaptation >= period AND epoch is divisible by period,
            False otherwise.
        """
        if state.last_adaptation_epoch < 0:
            # No prior adaptation - use epoch modulo
            return state.epoch % self.period == 0 and state.epoch > 0
        else:
            # Fire when period epochs have passed since last adaptation
            return epochs_since_adaptation(state) >= self.period


class ScheduledTrigger(BaseTrigger):
    """
    Trigger that fires at specific pre-defined epochs.
    
    This trigger fires only at epochs that are in a pre-defined set. It's useful
    for replicating manual adaptation schedules or when you have prior knowledge
    about good adaptation points.
    
    This is equivalent to the current manual approach:
        grid_upds = {0: 3, 200: 6, 400: 10, 600: 24}
    
    Attributes
    ----------
    epochs : Set[int]
        Set of epochs at which the trigger should fire.
    cooldown : int
        Inherited from BaseTrigger. Minimum epochs between adaptations.
    
    Notes
    -----
    The trigger only checks membership in the epochs set; it doesn't care about
    the order or spacing of the epochs. If an epoch in the set is skipped due
    to cooldown, it will not be triggered later.
    
    Examples
    --------
    >>> trigger = ScheduledTrigger(epochs=[0, 100, 250, 500])
    >>> 
    >>> state = AdaptationState(epoch=0)
    >>> print(trigger.should_adapt(state))  # True
    >>> 
    >>> state = AdaptationState(epoch=50)
    >>> print(trigger.should_adapt(state))  # False
    >>> 
    >>> state = AdaptationState(epoch=100)
    >>> print(trigger.should_adapt(state))  # True
    """
    
    def __init__(self, epochs: List[int], cooldown: int = 0):
        """
        Initialize the scheduled trigger.
        
        Parameters
        ----------
        epochs : List[int]
            List of epochs at which the trigger should fire. Will be converted
            to a set for O(1) lookup. Duplicate epochs are automatically removed.
            Can be empty (will never trigger).
        cooldown : int, optional
            Minimum epochs between adaptations. Default is 0.
        
        Examples
        --------
        >>> # Replicate manual schedule
        >>> trigger = ScheduledTrigger(epochs=[0, 200, 400, 600])
        
        >>> # With cooldown (epochs too close together will be skipped)
        >>> trigger = ScheduledTrigger(epochs=[0, 50, 100, 200], cooldown=75)
        >>> # Epoch 50 would be skipped due to cooldown from epoch 0
        """
        super().__init__(cooldown)
        self.epochs: Set[int] = set(epochs) if epochs else set()
    
    def should_trigger(self, state: AdaptationState) -> bool:
        """
        Check if current epoch is in the scheduled set.
        
        Parameters
        ----------
        state : AdaptationState
            Current adaptation state.
        
        Returns
        -------
        bool
            True if current epoch is in the scheduled set, False otherwise.
        """
        return state.epoch in self.epochs


class LossPlateauTrigger(BaseTrigger):
    """
    Trigger that fires when training loss plateaus.
    
    This trigger monitors the training loss and fires when no significant
    improvement has been observed for a specified number of epochs (patience).
    It's the most sophisticated trigger, allowing adaptation to respond to
    actual training dynamics.
    
    A "plateau" is detected when:
    1. At least `patience` epochs have passed without significant improvement
    2. "Significant improvement" means: loss < best_loss * (1 - rel_threshold)
    
    The relative threshold ensures that the definition of "improvement" scales
    with the current loss magnitude. For example, with rel_threshold=0.05:
    - At loss=1.0, improvement requires loss < 0.95
    - At loss=0.001, improvement requires loss < 0.00095
    
    Attributes
    ----------
    patience : int
        Number of epochs without improvement before triggering.
    rel_threshold : float
        Relative improvement threshold (e.g., 0.05 = 5%).
    cooldown : int
        Inherited from BaseTrigger. Minimum epochs between adaptations.
    
    Notes
    -----
    The state tracking (best_loss, epochs_since_improvement) is done by the
    `update_state()` function in state.py, which should be called every epoch.
    This trigger only reads the state; it doesn't modify it.
    
    After adaptation, call `reset_after_adaptation()` to reset the plateau
    tracking and begin monitoring for the next plateau.
    
    Implementation Detail
    ---------------------
    The rel_threshold in this trigger should match the rel_threshold used in
    update_state() for consistent behavior. Both default to 0.05 (5%).
    
    Examples
    --------
    >>> trigger = LossPlateauTrigger(patience=50, rel_threshold=0.05)
    >>> 
    >>> # Loss hasn't improved for 50 epochs -> trigger fires
    >>> state = AdaptationState(epochs_since_improvement=50)
    >>> print(trigger(state))  # True
    >>> 
    >>> # Loss still improving
    >>> state = AdaptationState(epochs_since_improvement=10)
    >>> print(trigger(state))  # False
    
    >>> # With cooldown to prevent too-frequent adaptations
    >>> trigger = LossPlateauTrigger(patience=50, rel_threshold=0.05, cooldown=100)
    """
    
    def __init__(
        self, 
        patience: int, 
        rel_threshold: float = 0.05, 
        cooldown: int = 0
    ):
        """
        Initialize the loss plateau trigger.
        
        Parameters
        ----------
        patience : int
            Number of epochs without significant improvement before triggering.
            Must be non-negative. Typical values range from 20 to 100 depending on
            the problem and training dynamics. A patience of 0 will trigger
            immediately when epochs_since_improvement >= 0.
        rel_threshold : float, optional
            Relative improvement threshold. Default is 0.05 (5%).
            This should match the rel_threshold used in update_state().
            - Higher values (e.g., 0.1) = more lenient, fewer triggers
            - Lower values (e.g., 0.01) = stricter, more frequent triggers
        cooldown : int, optional
            Minimum epochs between adaptations. Default is 0.
            Setting cooldown >= patience effectively ensures at least one
            "fair" training period after each adaptation.
        
        Raises
        ------
        ValueError
            If patience < 0, rel_threshold <= 0, rel_threshold >= 1, or cooldown < 0.
        
        Examples
        --------
        >>> # Standard configuration
        >>> trigger = LossPlateauTrigger(patience=50)
        
        >>> # Stricter improvement requirement
        >>> trigger = LossPlateauTrigger(patience=30, rel_threshold=0.01)
        
        >>> # With cooldown equal to patience
        >>> trigger = LossPlateauTrigger(patience=50, cooldown=50)
        """
        super().__init__(cooldown)
        if patience < 0:
            raise ValueError(f"patience must be non-negative, got {patience}")
        if rel_threshold <= 0 or rel_threshold >= 1:
            raise ValueError(
                f"rel_threshold must be in (0, 1), got {rel_threshold}"
            )
        self.patience = patience
        self.rel_threshold = rel_threshold
    
    def should_trigger(self, state: AdaptationState) -> bool:
        """
        Check if a loss plateau has been detected.
        
        Parameters
        ----------
        state : AdaptationState
            Current adaptation state. Must have been updated with update_state()
            each epoch for accurate plateau detection.
        
        Returns
        -------
        bool
            True if epochs_since_improvement >= patience, False otherwise.
        
        Notes
        -----
        The actual improvement tracking is done by update_state(). This method
        simply checks if the patience threshold has been reached.
        """
        return state.epochs_since_improvement >= self.patience


class CompositeTrigger(BaseTrigger):
    """
    Trigger that combines multiple triggers with configurable logic.
    
    This trigger allows you to combine several triggers and specify how they
    should be evaluated together. You can require ANY of them to fire (OR logic)
    or ALL of them to fire (AND logic).
    
    Common Use Cases
    ----------------
    1. **Scheduled + Loss-based**: Adapt at specific epochs OR when loss plateaus.
       This ensures adaptation happens at known good points, but also responds
       to training dynamics.
    
    2. **Periodic + Loss-based with AND**: Only adapt periodically IF loss has
       also plateaued. This prevents unnecessary adaptations when training is
       progressing well.
    
    3. **Multiple schedules**: Combine different scheduled triggers with OR logic.
    
    Attributes
    ----------
    triggers : List[BaseTrigger]
        List of triggers to combine.
    mode : str
        Combination mode: "any" (OR logic) or "all" (AND logic).
    cooldown : int
        Inherited from BaseTrigger. Applied to the composite trigger itself,
        independent of individual trigger cooldowns.
    
    Notes
    -----
    - Each sub-trigger maintains its own cooldown (if any).
    - The composite trigger also has its own cooldown, applied after any
      sub-trigger condition is met.
    - With mode="any", the composite fires if ANY sub-trigger fires.
    - With mode="all", the composite fires only if ALL sub-triggers fire.
    
    Examples
    --------
    >>> # Adapt at scheduled epochs OR when loss plateaus
    >>> scheduled = ScheduledTrigger(epochs=[100, 200, 300])
    >>> plateau = LossPlateauTrigger(patience=50)
    >>> trigger = CompositeTrigger([scheduled, plateau], mode="any")
    
    >>> # Adapt periodically, but only if loss has also plateaued
    >>> periodic = PeriodicTrigger(period=100)
    >>> plateau = LossPlateauTrigger(patience=30)
    >>> trigger = CompositeTrigger([periodic, plateau], mode="all")
    
    >>> # Complex: scheduled epochs OR (periodic AND plateaued)
    >>> scheduled = ScheduledTrigger(epochs=[0, 500])
    >>> periodic_and_plateau = CompositeTrigger(
    ...     [PeriodicTrigger(period=100), LossPlateauTrigger(patience=30)],
    ...     mode="all"
    ... )
    >>> trigger = CompositeTrigger([scheduled, periodic_and_plateau], mode="any")
    """
    
    def __init__(
        self, 
        triggers: List[BaseTrigger], 
        mode: str = "any",
        cooldown: int = 0
    ):
        """
        Initialize the composite trigger.
        
        Parameters
        ----------
        triggers : List[BaseTrigger]
            List of triggers to combine. Must contain at least one trigger.
            Can include other CompositeTriggers for nested logic.
        mode : str, optional
            How to combine trigger results. Default is "any".
            - "any": Fire if ANY sub-trigger fires (OR logic)
            - "all": Fire only if ALL sub-triggers fire (AND logic)
        cooldown : int, optional
            Minimum epochs between composite trigger activations. Default is 0.
            This is independent of individual trigger cooldowns.
        
        Raises
        ------
        ValueError
            If triggers list is empty or mode is not "any" or "all".
        
        Examples
        --------
        >>> # OR logic: adapt at scheduled epochs or when loss plateaus
        >>> trigger = CompositeTrigger([
        ...     ScheduledTrigger(epochs=[100, 200]),
        ...     LossPlateauTrigger(patience=50)
        ... ], mode="any")
        
        >>> # AND logic: adapt only when both conditions are met
        >>> trigger = CompositeTrigger([
        ...     PeriodicTrigger(period=100),
        ...     LossPlateauTrigger(patience=30)
        ... ], mode="all")
        """
        super().__init__(cooldown)
        
        if not triggers:
            raise ValueError("triggers list must not be empty")
        
        if mode not in ("any", "all"):
            raise ValueError(f"mode must be 'any' or 'all', got '{mode}'")
        
        self.triggers = triggers
        self.mode = mode
    
    def should_trigger(self, state: AdaptationState) -> bool:
        """
        Check if the composite trigger condition is met.
        
        Parameters
        ----------
        state : AdaptationState
            Current adaptation state.
        
        Returns
        -------
        bool
            If mode="any": True if ANY sub-trigger fires.
            If mode="all": True only if ALL sub-triggers fire.
        
        Notes
        -----
        Each sub-trigger's __call__ method is invoked, which includes
        their individual cooldown checks. The composite's own cooldown
        is checked by the base class before this method is called.
        """
        if self.mode == "any":
            return any(trigger(state) for trigger in self.triggers)
        else:  # mode == "all"
            return all(trigger(state) for trigger in self.triggers)
    
    def __repr__(self) -> str:
        """Return a string representation of the composite trigger."""
        trigger_strs = [type(t).__name__ for t in self.triggers]
        return f"CompositeTrigger(triggers=[{', '.join(trigger_strs)}], mode='{self.mode}')"
