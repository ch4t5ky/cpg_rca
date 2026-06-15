"""
cpg.constants
~~~~~~~~~~~~~
CPG node / edge label strings that match the Joern DOT export format,
and project-level defaults.
"""

# Node / edge label values
LABEL_METHOD   = "METHOD"
LABEL_CALL     = "CALL"
EDGE_CONTAINS  = "CONTAINS"

# Node attribute keys
ATTR_LABEL           = "label"
ATTR_NAME            = "NAME"
ATTR_FULLNAME        = "FULLNAME"
ATTR_FILENAME        = "FILENAME"
ATTR_CODE            = "CODE"
ATTR_IS_EXTERNAL     = "IS_EXTERNAL"
ATTR_METHOD_FULLNAME = "METHODFULLNAME"

# Default FULLNAME prefixes that mark a node as owned by the service.
# Override by passing project_prefixes=(...) to CodePropertyGraph.
DEFAULT_PROJECT_PREFIXES: tuple[str, ...] = (
    "main.",
    "main",
    "github.com/GoogleCloudPlatform/microservices-demo/src/frontend",
)
