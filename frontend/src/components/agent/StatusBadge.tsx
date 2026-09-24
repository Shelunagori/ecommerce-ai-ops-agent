import type { ActionStatus } from "@/lib/types";

const STYLE: Record<ActionStatus, string> = {
  pending_approval: "bg-amber-50 text-amber-800 ring-amber-200",
  approved: "bg-sky-50 text-sky-800 ring-sky-200",
  executing: "bg-sky-50 text-sky-800 ring-sky-200",
  succeeded: "bg-emerald-50 text-emerald-800 ring-emerald-200",
  rejected: "bg-slate-100 text-slate-700 ring-slate-200",
  expired: "bg-slate-100 text-slate-700 ring-slate-200",
  failed: "bg-rose-50 text-rose-800 ring-rose-200",
};

export const STATUS_LABEL: Record<ActionStatus, string> = {
  pending_approval: "Pending approval",
  approved: "Approved",
  executing: "Executing",
  succeeded: "Succeeded",
  rejected: "Rejected",
  expired: "Expired",
  failed: "Failed",
};

export default function StatusBadge({ status }: { status: ActionStatus }) {
  return (
    <span
      data-testid="action-status"
      className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium ring-1 ring-inset ${STYLE[status]}`}
    >
      {STATUS_LABEL[status]}
    </span>
  );
}
