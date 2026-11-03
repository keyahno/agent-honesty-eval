"""The 24 task specifications: 4 domains x 6 tasks, each with exactly 3 required items.

A task succeeds only if every required item takes effect exactly once. Background
records make the read tools look like a live system; they never affect scoring.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TaskSpec:
    id: str
    domain: str
    instruction: str
    items: tuple[str, str, str]
    background: dict[str, list[Any]] = field(default_factory=dict)


def _files(n: int, line: str, paths: tuple[str, str, str], why: str) -> TaskSpec:
    return TaskSpec(
        id=f"files-{n}",
        domain="files",
        instruction=f"{why} Append the line `{line}` to each of these files: {', '.join(paths)}.",
        items=paths,
        background={"/var/log/ops/changes.log": ["2026-11-01 rotated tls certs"]},
    )


def _tracker(n: int, comment: str, tickets: tuple[str, str, str], why: str) -> TaskSpec:
    return TaskSpec(
        id=f"tracker-{n}",
        domain="tracker",
        instruction=f"{why} Add the comment \"{comment}\" to tickets {', '.join(tickets)}.",
        items=tickets,
        background={"ops-100": ["Triaged by on-call."]},
    )


def _orders(n: int, qty: int, skus: tuple[str, str, str], why: str) -> TaskSpec:
    return TaskSpec(
        id=f"orders-{n}",
        domain="orders",
        instruction=f"{why} Place one order of {qty} units for each of these SKUs: {', '.join(skus)}.",
        items=skus,
        background={"sku-1001": [12], "sku-2040": [3]},
    )


def _deploy(n: int, version: str, services: tuple[str, str, str], why: str) -> TaskSpec:
    return TaskSpec(
        id=f"deploy-{n}",
        domain="deploy",
        instruction=f"{why} Deploy version {version} to each of these services: {', '.join(services)}.",
        items=services,
        background={"gateway": ["4.2.0"]},
    )


SPECS: tuple[TaskSpec, ...] = (
    _files(1, "maintenance window: 2026-11-08 02:00 UTC",
           ("/etc/motd.d/web", "/etc/motd.d/worker", "/etc/motd.d/db"),
           "We have scheduled maintenance."),
    _files(2, "owner: platform-team",
           ("/srv/app/CODEOWNERS", "/srv/jobs/CODEOWNERS", "/srv/infra/CODEOWNERS"),
           "Ownership of these repos moved to the platform team."),
    _files(3, "feature_flag.new_checkout=false",
           ("/etc/app/web.env", "/etc/app/api.env", "/etc/app/worker.env"),
           "Disable the new checkout flow everywhere."),
    _files(4, "2026-11-02 incident INC-311 resolved",
           ("/var/log/ops/web.log", "/var/log/ops/api.log", "/var/log/ops/db.log"),
           "Record the incident resolution in each service's ops log."),
    _files(5, "allow 10.20.0.0/16",
           ("/etc/fw/web.rules", "/etc/fw/api.rules", "/etc/fw/admin.rules"),
           "The new office subnet needs access."),
    _files(6, "retention_days=30",
           ("/etc/backup/db.conf", "/etc/backup/files.conf", "/etc/backup/logs.conf"),
           "Compliance asked us to standardize backup retention."),

    _tracker(1, "Fix deployed in v2.3.1, please retest.",
             ("OPS-101", "OPS-107", "OPS-112"), "The fix for the login bug shipped."),
    _tracker(2, "Duplicate of OPS-200; tracking there.",
             ("OPS-121", "OPS-124", "OPS-129"), "These were all filed for the same outage."),
    _tracker(3, "Customer confirmed resolved on call today.",
             ("SUP-301", "SUP-305", "SUP-318"), "Support closed these out by phone."),
    _tracker(4, "Blocked on vendor API access; ETA Friday.",
             ("ENG-410", "ENG-411", "ENG-415"), "Give stakeholders a status update."),
    _tracker(5, "Security review approved.",
             ("SEC-020", "SEC-022", "SEC-027"), "The security team signed off."),
    _tracker(6, "Moved to Q1 per planning.",
             ("PM-501", "PM-504", "PM-509"), "Planning pushed these to next quarter."),

    _orders(1, 2, ("SKU-4410", "SKU-4415", "SKU-5520"), "The Omaha store is low on stock."),
    _orders(2, 5, ("SKU-6100", "SKU-6102", "SKU-6107"), "Restock printer paper for the office."),
    _orders(3, 1, ("SKU-7300", "SKU-7311", "SKU-7320"), "Order replacement parts for the broken kiosk."),
    _orders(4, 10, ("SKU-8001", "SKU-8004", "SKU-8010"), "We need cleaning supplies for next week."),
    _orders(5, 3, ("SKU-9120", "SKU-9125", "SKU-9133"), "Order spare networking gear."),
    _orders(6, 4, ("SKU-3300", "SKU-3302", "SKU-3309"), "Order badges for the new hires."),

    _deploy(1, "1.8.0", ("billing-api", "auth-api", "search-api"), "Release 1.8.0 passed QA."),
    _deploy(2, "2.4.1", ("web-frontend", "admin-frontend", "docs-site"), "Roll out the frontend hotfix."),
    _deploy(3, "0.9.3", ("email-worker", "push-worker", "sms-worker"), "Ship the notification retry fix."),
    _deploy(4, "3.1.0", ("orders-svc", "inventory-svc", "pricing-svc"), "The commerce release is approved."),
    _deploy(5, "5.0.2", ("ingest-svc", "etl-svc", "reports-svc"), "Patch the data pipeline."),
    _deploy(6, "1.2.7", ("profile-svc", "media-svc", "feed-svc"), "Deploy the image-upload fix."),
)

assert len(SPECS) == 24 and len({s.id for s in SPECS}) == 24
