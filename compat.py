import inspect
import re

from torch_geometric.nn.conv import gcn_conv

source = inspect.getsource(gcn_conv.gcn_norm)
fixed = source.replace("deg.pow_(-0.5)", "deg.float().pow(-0.5)")
namespace = gcn_conv.__dict__.copy()
exec(compile(fixed, "<compat>", "exec"), namespace)
gcn_conv.gcn_norm = namespace["gcn_norm"]