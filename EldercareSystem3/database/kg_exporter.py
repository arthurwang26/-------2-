import sys
import json
from pathlib import Path
from typing import List, Dict, Any

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import cfg
from utils.logger import get_logger

logger = get_logger("kg_exporter")


class KnowledgeGraphExporter:
    """Exports structured events into a Knowledge Graph format (JSON-LD inspired)."""

    def __init__(self):
        self.nodes = {}
        self.edges = []
        self.node_id_counter = 1
        
        # Keep track of label to ID mapping to avoid duplicates
        self._label_to_id = {}

    def _get_or_create_node(self, label: str, node_type: str) -> str:
        key = f"{node_type}::{label}"
        if key in self._label_to_id:
            return self._label_to_id[key]
        
        node_id = f"n{self.node_id_counter}"
        self.node_id_counter += 1
        
        self.nodes[node_id] = {
            "id": node_id,
            "label": label,
            "type": node_type
        }
        self._label_to_id[key] = node_id
        return node_id

    def _add_edge(self, source_id: str, target_id: str, relation: str, properties: Dict = None):
        if properties is None:
            properties = {}
            
        edge = {
            "source": source_id,
            "target": target_id,
            "relation": relation,
            "properties": properties
        }
        # Prevent exact duplicate edges
        if edge not in self.edges:
            self.edges.append(edge)

    def process_events(self, events_by_clip: Dict[str, List[Dict]]):
        """Build the graph from clip events."""
        logger.info("Building Knowledge Graph from events...")
        
        for clip_name, events in events_by_clip.items():
            clip_node = self._get_or_create_node(clip_name, "VideoClip")
            
            for ev in events:
                person_name = ev.get("person", "Unknown")
                if person_name == "Unknown":
                    continue
                    
                person_node = self._get_or_create_node(person_name, "Person")
                self._add_edge(clip_node, person_node, "CONTAINS")
                
                event_type = ev.get("type", "")
                conf = ev.get("confidence", 1.0)
                
                if event_type == "Action":
                    action_label = ev.get("action")
                    action_node = self._get_or_create_node(action_label, "Action")
                    self._add_edge(person_node, action_node, "PERFORMS_ACTION", {"confidence": conf})
                    
                elif event_type == "Emotion":
                    emotion_label = ev.get("emotion")
                    emotion_node = self._get_or_create_node(emotion_label, "Emotion")
                    self._add_edge(person_node, emotion_node, "FEELS_EMOTION", {"confidence": conf})
                    
                elif event_type.startswith("HOI"):
                    action_label = ev.get("action")
                    object_label = ev.get("object")
                    
                    if action_label and object_label:
                        action_node = self._get_or_create_node(action_label, "Action")
                        object_node = self._get_or_create_node(object_label, "Object")
                        
                        # Person performs action
                        self._add_edge(person_node, action_node, "PERFORMS_ACTION", {"confidence": conf})
                        # Action target is Object
                        self._add_edge(action_node, object_node, "TARGETS_OBJECT", {"confidence": conf})
                        # Direct Person -> Object interaction
                        self._add_edge(person_node, object_node, "INTERACTS_WITH", {"action": action_label, "confidence": conf})

    def export(self):
        """Export the constructed graph to a JSON file."""
        base_dir = cfg.output_dir / "database"
        base_dir.mkdir(parents=True, exist_ok=True)
        
        kg_path = base_dir / "knowledge_graph.json"
        
        graph_data = {
            "nodes": list(self.nodes.values()),
            "edges": self.edges
        }
        
        try:
            with open(kg_path, "w", encoding="utf-8") as f:
                json.dump(graph_data, f, ensure_ascii=False, indent=2)
            logger.info(f"Knowledge Graph exported to {kg_path} ({len(self.nodes)} nodes, {len(self.edges)} edges)")
        except Exception as e:
            logger.error(f"Failed to export Knowledge Graph: {e}")
