from . import expert_pruning
from . import dynamic_skipping
from . import nas_pruning

METHODS = {
    'layerwise_pruning': expert_pruning.layerwise_pruning,
    'progressive_pruning': expert_pruning.progressive_pruning,
    'nas_pruning': nas_pruning.nas_pruning,
    'dynamic_skipping': dynamic_skipping.dynamic_skipping,
    'load_pruning': nas_pruning.pruning_model_from_load
}