"""
Adaptation State Management
===========================

This module provides the state management infrastructure for the grid adaptation
framework. It defines the state dataclass and utility functions for updating
state during the training loop.

Design Philosophy
-----------------
The state is designed to be:
1. **Lightweight**: Only store scalar values and small counters to minimize overhead.
2. **Immutable-style**: Use dataclass `replace()` to create new state instances,
   making the code functional and easier to reason about.
3. **JIT-friendly**: All state fields are simple Python/JAX types that work with JIT.
4. **Extensible**: The dataclass can be extended with additional fields for future
   trigger types or per-layer tracking.

Usage Pattern
-------------
The typical usage pattern in a training loop is:

    state = AdaptationState()
    
    for epoch in range(num_epochs):
        loss = train_step(...)
        state = update_state(state, loss)
        
        if trigger(state):
            model.adapt(...)
            state = reset_after_adaptation(state)

This keeps the state management outside the JIT-compiled training step for flexibility,
while the trigger check itself is O(1) and can be JIT-compiled if needed.
"""

from dataclasses import dataclass, replace
from typing import Optional
import jax.numpy as jnp


@dataclass
class AdaptationState:
    """
    Lightweight state container for tracking adaptation triggers.
    
    This dataclass holds all the information needed by triggers to decide
    whether an adaptation should occur. It is updated once per epoch with
    minimal computational overhead.
    
    Attributes
    ----------
    epoch : int
        Current epoch number (0-indexed). Incremented by update_state().
    best_loss : float
        Best (lowest) loss value seen since the last adaptation. Used by
        plateau-based triggers to detect when training has stalled.
    epochs_since_improvement : int
        Number of consecutive epochs without significant improvement.
        "Significant" is defined by the rel_threshold parameter in update_state().
        Reset to 0 when improvement occurs or after adaptation.
    last_adaptation_epoch : int
        Epoch number when the last adaptation occurred. Set to -1 initially
        (no adaptation has occurred). Used for cooldown calculations.
    
    Notes
    -----
    - All fields are simple scalar types for minimal memory and JIT compatibility.
    - The state is designed to be replaced (not mutated) using dataclass replace().
    - Future extensions might add fields like `adaptation_count`, `loss_history`, etc.
    
    Examples
    --------
    >>> # Create initial state
    >>> state = AdaptationState()
    >>> print(state.epoch)
    0
    
    >>> # State after some training
    >>> state = AdaptationState(
    ...     epoch=100,
    ...     best_loss=0.05,
    ...     epochs_since_improvement=10,
    ...     last_adaptation_epoch=50
    ... )
    """
    epoch: int = 0
    best_loss: float = float('inf')
    epochs_since_improvement: int = 0
    last_adaptation_epoch: int = -1


def update_state(
    state: AdaptationState, 
    loss: float, 
    rel_threshold: float = 0.05
) -> AdaptationState:
    """
    Update the adaptation state after completing an epoch.
    
    This function should be called once per epoch, after computing the epoch's
    loss. It tracks whether the loss has improved significantly and updates
    the relevant counters.
    
    Parameters
    ----------
    state : AdaptationState
        Current state before this epoch's update.
    loss : float
        The loss value for the current epoch. This should be the total/average
        loss over the training data, not per-sample loss.
    rel_threshold : float, optional
        Relative improvement threshold. Default is 0.05 (5%).
        The loss is considered to have "improved" only if:
            loss < best_loss * (1 - rel_threshold)
        
        For example, with rel_threshold=0.05:
        - If best_loss=1.0, improvement requires loss < 0.95
        - If best_loss=0.1, improvement requires loss < 0.095
        - If best_loss=0.01, improvement requires loss < 0.0095
        
        This relative threshold ensures that "improvement" scales with the
        current loss magnitude, avoiding premature triggering on noise.
    
    Returns
    -------
    AdaptationState
        New state with updated fields:
        - epoch: incremented by 1
        - best_loss: updated if significant improvement occurred
        - epochs_since_improvement: reset to 0 if improved, else incremented
        - last_adaptation_epoch: unchanged (only modified by reset_after_adaptation)
    
    Notes
    -----
    This function is O(1) and performs only scalar comparisons, making it
    suitable for calling every epoch without performance concerns.
    
    The function uses relative threshold rather than absolute threshold because:
    1. Loss magnitudes vary widely across problems (1e-2 to 1e-8)
    2. What constitutes "noise" is proportional to the current loss scale
    3. Early training (high loss) and late training (low loss) need different sensitivities
    
    Examples
    --------
    >>> state = AdaptationState(epoch=10, best_loss=0.1, epochs_since_improvement=0)
    
    >>> # Significant improvement (> 5% decrease)
    >>> new_state = update_state(state, loss=0.08)
    >>> print(new_state.best_loss)
    0.08
    >>> print(new_state.epochs_since_improvement)
    0
    
    >>> # No significant improvement (< 5% decrease)
    >>> new_state = update_state(state, loss=0.098)
    >>> print(new_state.best_loss)
    0.1  # unchanged
    >>> print(new_state.epochs_since_improvement)
    1
    
    >>> # Loss increased
    >>> new_state = update_state(state, loss=0.15)
    >>> print(new_state.epochs_since_improvement)
    1
    """
    new_epoch = state.epoch + 1
    
    # Calculate the threshold for "significant improvement"
    # Loss must decrease by at least rel_threshold fraction of best_loss
    improvement_threshold = state.best_loss * (1.0 - rel_threshold)
    
    if loss < improvement_threshold:
        # Significant improvement occurred
        return replace(
            state,
            epoch=new_epoch,
            best_loss=loss,
            epochs_since_improvement=0
        )
    else:
        # No significant improvement
        # Note: best_loss is only updated on significant improvement
        # This prevents slow drift from resetting the plateau counter
        return replace(
            state,
            epoch=new_epoch,
            epochs_since_improvement=state.epochs_since_improvement + 1
        )


def reset_after_adaptation(state: AdaptationState) -> AdaptationState:
    """
    Reset the state after an adaptation has been performed.
    
    This function should be called immediately after performing a grid adaptation.
    It resets the tracking counters to begin monitoring for the next plateau,
    and records when the adaptation occurred for cooldown calculations.
    
    Parameters
    ----------
    state : AdaptationState
        Current state at the time of adaptation.
    
    Returns
    -------
    AdaptationState
        New state with:
        - epoch: unchanged (still the current epoch)
        - best_loss: reset to infinity (start fresh tracking)
        - epochs_since_improvement: reset to 0
        - last_adaptation_epoch: set to current epoch
    
    Notes
    -----
    Resetting best_loss to infinity ensures that the first epoch after adaptation
    will always count as an "improvement", giving the adapted model a chance to
    show its performance before plateau detection kicks in.
    
    The last_adaptation_epoch is used by triggers that implement cooldown periods
    to prevent rapid successive adaptations.
    
    Examples
    --------
    >>> state = AdaptationState(
    ...     epoch=100,
    ...     best_loss=0.05,
    ...     epochs_since_improvement=50,
    ...     last_adaptation_epoch=20
    ... )
    >>> new_state = reset_after_adaptation(state)
    >>> print(new_state.epoch)
    100
    >>> print(new_state.best_loss)
    inf
    >>> print(new_state.epochs_since_improvement)
    0
    >>> print(new_state.last_adaptation_epoch)
    100
    """
    return replace(
        state,
        best_loss=float('inf'),
        epochs_since_improvement=0,
        last_adaptation_epoch=state.epoch
    )


def epochs_since_adaptation(state: AdaptationState) -> int:
    """
    Calculate the number of epochs since the last adaptation.
    
    This utility function is useful for triggers that implement cooldown periods.
    
    Parameters
    ----------
    state : AdaptationState
        Current adaptation state.
    
    Returns
    -------
    int
        Number of epochs since the last adaptation. Returns the current epoch
        if no adaptation has occurred yet (last_adaptation_epoch == -1).
    
    Examples
    --------
    >>> state = AdaptationState(epoch=100, last_adaptation_epoch=80)
    >>> print(epochs_since_adaptation(state))
    20
    
    >>> # No adaptation yet
    >>> state = AdaptationState(epoch=50, last_adaptation_epoch=-1)
    >>> print(epochs_since_adaptation(state))
    50
    """
    if state.last_adaptation_epoch < 0:
        return state.epoch
    return state.epoch - state.last_adaptation_epoch
