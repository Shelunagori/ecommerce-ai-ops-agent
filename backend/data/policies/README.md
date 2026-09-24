# Synthetic policy corpus (demo only)

Every document here is **fictional**, written for the CommerceOps AI demo. The two tenants
(Northstar Commerce, BluePeak Retail) are invented; none of this is a real company's policy.

Layout: `<tenant-slug>/<document_key>.v<version>.md`, each file starting with a YAML front
matter block (`tenant`, `document_key`, `title`, `document_type`, `version`,
`effective_from`, `effective_to`). `effective_to` is **exclusive**: a version applies when
`effective_from <= as_of < effective_to` (UTC calendar dates). Versions are immutable: a
change is a new file with a higher version, never an edit of an ingested one.
