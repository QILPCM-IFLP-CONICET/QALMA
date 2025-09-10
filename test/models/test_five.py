from qalma.alpsmodels import model_from_alps_xml
from qalma.geometry import graph_from_alps_xml
from qalma.model import SystemDescriptor

latt_descr = graph_from_alps_xml(name="five-open", parms={"L": 4, "a": 1})
model_descr = model_from_alps_xml(name="spin")
SYSTEM = SystemDescriptor(latt_descr, model_descr, {})
