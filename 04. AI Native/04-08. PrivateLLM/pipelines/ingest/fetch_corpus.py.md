
```
from ..config import Config  
from .law_client import LawClient  
from .law_parser import parse_law  
from .prec_parser import parse_prec

def collect(*, client, out_dir: Path, law_queries=None, prec_queries=None) ->    None:  
	parse_law  
	parse_prec
	
def _match_law_mst(search_xml: str, query: str) -> str | None:

def _write(path: Path, obj: dict) -> None:

def _all(xml_text: str, tag: str) -> list[str]:

def _slug(s: str) -> str:

def main() -> None:

	cfg = Config.from_env()

	collect(client=LawClient(oc=cfg.oc), out_dir=cfg.raw_dir)
```
