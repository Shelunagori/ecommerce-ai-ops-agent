"""Stable, safe action errors (codes are part of the API/graph contract).

Messages never contain SQL, stack traces, raw exception text or another tenant's data. An
action request of another tenant is reported exactly like a missing one.
"""

from app.core.errors import AppError


class ActionError(AppError):
    status_code = 409
    code = "action_error"
    message = "The action could not be completed."

    def __init__(self, message: str | None = None, *, detail: str | None = None) -> None:
        self.detail = detail  # safe internal classification (e.g. current order status)
        super().__init__(message)


class ActionArgumentsInvalidError(ActionError):
    status_code = 400
    code = "action_invalid_arguments"
    message = "The action arguments are invalid."


class ActionNotFoundError(ActionError):
    status_code = 404
    code = "action_not_found"
    message = "The action request was not found."


class ActionTargetNotFoundError(ActionError):
    status_code = 404

    def __init__(self, resource: str, reference: str) -> None:
        self.code = f"{resource}_not_found"
        super().__init__(f"{resource.capitalize()} '{reference}' was not found.")


class OrderNotCancellableError(ActionError):
    code = "order_not_cancellable"

    def __init__(self, order_number: str, status: str) -> None:
        super().__init__(
            f"Order '{order_number}' cannot be cancelled in status '{status}'.", detail=status
        )


class StoreCreditRejectedError(ActionError):
    status_code = 400
    code = "store_credit_rejected"
    message = "The store credit request does not satisfy the credit rules."


class EvidenceRequiredError(ActionError):
    status_code = 400
    code = "evidence_required"
    message = "This action requires policy evidence retrieved in the current request."


class DuplicateOpenActionError(ActionError):
    code = "duplicate_open_action"
    message = "An open action request already exists for this target."


class IdempotencyConflictError(ActionError):
    code = "idempotency_conflict"
    message = "The idempotency key was already used with different arguments."


class ApprovalExpiredError(ActionError):
    code = "approval_expired"
    message = "The approval window for this action has expired; nothing was executed."


class ApprovalResolvedError(ActionError):
    code = "approval_already_resolved"
    message = "This action request was already resolved."


class ArgumentsHashMismatchError(ActionError):
    code = "arguments_hash_mismatch"
    message = "The action changed since it was shown for approval; a new approval is required."


class ActionNotApprovedError(ActionError):
    code = "action_not_approved"
    message = "The action has not been approved; nothing was executed."


class ActionInProgressError(ActionError):
    code = "action_in_progress"
    message = "The action is currently being executed."


class ActionExecutionFailedError(ActionError):
    status_code = 500
    code = "action_execution_failed"
    message = "The action could not be executed; nothing was changed."


class ActionForbiddenError(ActionError):
    status_code = 403
    code = "action_forbidden"
    message = "You are not allowed to decide on this action."
