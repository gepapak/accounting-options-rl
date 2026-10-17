"""Static AST inventory of frozen native state/randomness; no native imports."""
from __future__ import annotations
import ast
import hashlib
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
ENGINE=HERE.parents[3]/"Prototype5"

def name(node):
    if isinstance(node,ast.Name):return node.id
    if isinstance(node,ast.Attribute):return name(node.value)+"."+node.attr
    if isinstance(node,ast.Call):return name(node.func)+"()"
    return type(node).__name__

def main():
    frozen=json.loads((HERE/"execution/run_001/SOURCE_FREEZE.json").read_text(encoding="utf-8"))
    output={"scope":"AST source inventory only; no native imports, actor calls, episodes or endpoints", "native_modules":34,"random_calls":[],"global_declarations":[],"class_or_module_mutable_assignments":[],"memory_manager_calls":[]}
    for file,digest in frozen["native_source_sha256"].items():
        source=(ENGINE/file).read_bytes();assert hashlib.sha256(source).hexdigest()==digest
        tree=ast.parse(source.decode("utf-8-sig"));parents={child:parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
        def owner(node):
            names=[];parent=parents.get(node)
            while parent is not None:
                if isinstance(parent,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):names.append(parent.name)
                parent=parents.get(parent)
            return ".".join(reversed(names)) or "<module>"
        for node in ast.walk(tree):
            if isinstance(node,ast.Call):
                call=name(node.func)
                if any(mark in call for mark in ("random.","_rng.","manual_seed","set_seed","randn","randint","rand_like","multinomial",".sample",".rsample")):
                    output["random_calls"].append({"file":file,"line":node.lineno,"owner":owner(node),"call":call})
                if "memory_manager" in call:output["memory_manager_calls"].append({"file":file,"line":node.lineno,"owner":owner(node),"call":call})
            if isinstance(node,ast.Global):output["global_declarations"].append({"file":file,"line":node.lineno,"owner":owner(node),"names":node.names})
            if isinstance(node,(ast.Assign,ast.AnnAssign)):
                parent=parents.get(node)
                if isinstance(parent,(ast.Module,ast.ClassDef)):
                    value=node.value
                    if isinstance(value,(ast.List,ast.Dict,ast.Set,ast.ListComp,ast.DictComp,ast.SetComp,ast.Call)):
                        targets=node.targets if isinstance(node,ast.Assign) else [node.target]
                        output["class_or_module_mutable_assignments"].append({"file":file,"line":node.lineno,"owner":owner(node),"targets":[name(target) for target in targets],"value":name(value)})
    destination=HERE/"validation/NATIVE_STATE_STATIC_INVENTORY.json"
    with destination.open("x",encoding="utf-8") as stream:json.dump(output,stream,indent=2,sort_keys=True)
    print(json.dumps(output,indent=2))

if __name__=="__main__":main()
