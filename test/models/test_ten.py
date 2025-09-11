import tempfile

from qalma.alpsmodels import model_from_alps_xml
from qalma.geometry import graph_from_alps_xml
from qalma.model import SystemDescriptor
from qalma.settings import LATTICE_LIB_FILE


def build_ten_sites_lattice_file():
    """
    Create a temporary lattice library file with an extra lattice
    """

    doble_five = """
    <UNITCELL name="kagome-stripe-double" dimension="1" vertices="10">
    <VERTEX><COORDINATE>-.125   0</COORDINATE></VERTEX>
    <VERTEX><COORDINATE> 0     0.125</COORDINATE></VERTEX>
    <VERTEX><COORDINATE> 0.0   0.0</COORDINATE></VERTEX>
    <VERTEX><COORDINATE> 0.   -0.125</COORDINATE></VERTEX>
    <VERTEX><COORDINATE> 0.125  0.</COORDINATE></VERTEX>
    <VERTEX><COORDINATE>.375    0.5</COORDINATE></VERTEX>
    <VERTEX><COORDINATE> 0.5      0.625</COORDINATE></VERTEX>
    <VERTEX><COORDINATE> 0.5    0.5</COORDINATE></VERTEX>
    <VERTEX><COORDINATE> 0.5     0.375</COORDINATE></VERTEX>
    <VERTEX><COORDINATE> 0.625  0.5</COORDINATE></VERTEX>

    <!-- split iterations -->
    <!--fist block of 5-->
    <EDGE type="1"><SOURCE vertex="1" offset="0"/><TARGET vertex="3" offset="0"/></EDGE>
    <EDGE type="1"><SOURCE vertex="3" offset="0"/><TARGET vertex="5" offset="0"/></EDGE>

    <EDGE type="2"><SOURCE vertex="2" offset="0"/><TARGET vertex="3" offset="0"/></EDGE>
    <EDGE type="2"><SOURCE vertex="3" offset="0"/><TARGET vertex="4" offset="0"/></EDGE>

    <EDGE type="3"><SOURCE vertex="1" offset="0"/><TARGET vertex="2" offset="0"/></EDGE>
    <EDGE type="3"><SOURCE vertex="4" offset="0"/><TARGET vertex="5" offset="0"/></EDGE>
    <!--segundos 5-->
    <EDGE type="2"><SOURCE vertex="6" offset="0"/><TARGET vertex="8" offset="0"/></EDGE>
    <EDGE type="2"><SOURCE vertex="8" offset="0"/><TARGET vertex="10" offset="0"/></EDGE>
    <EDGE type="1"><SOURCE vertex="7" offset="0"/><TARGET vertex="8" offset="0"/></EDGE>
    <EDGE type="1"><SOURCE vertex="8" offset="0"/><TARGET vertex="9" offset="0"/></EDGE>
    <EDGE type="3"><SOURCE vertex="6" offset="0"/><TARGET vertex="7" offset="0"/></EDGE>
    <EDGE type="3"><SOURCE vertex="9" offset="0"/><TARGET vertex="10" offset="0"/></EDGE>
    <!--internal bridge-->
    <EDGE type="4"><SOURCE vertex="2" offset="0"/><TARGET vertex="6" offset="0"/></EDGE>
    <EDGE type="5"><SOURCE vertex="5" offset="0"/><TARGET vertex="9" offset="0"/></EDGE>
    <!--external bridge-->
    <EDGE type="4"><SOURCE vertex="10" offset="0"/><TARGET vertex="4" offset="1"/></EDGE>
    <EDGE type="5"><SOURCE vertex="7" offset="0"/><TARGET vertex="1" offset="1"/></EDGE>
    </UNITCELL>
    
    <LATTICEGRAPH name = "kagome-stripe-double">
    <FINITELATTICE>
    <LATTICE ref="chain lattice"/>
    <EXTENT dimension="1" size ="L"/>
    <BOUNDARY type="periodic"/>
    </FINITELATTICE>
    <UNITCELL ref="kagome-stripe-double"/>
    </LATTICEGRAPH>
    """
    with open(LATTICE_LIB_FILE, "r") as f:
        lines = f.readlines()
        lines = lines[:-2] + [""]

    with tempfile.NamedTemporaryFile(suffix=".xml", mode="w", delete=False) as fp:
        fp.write("".join(lines))
        fp.write(doble_five)
        fp.write("</LATTICES>")
    return fp.name


def test_complex_model():
    latt_descr = graph_from_alps_xml(
        filename=build_ten_sites_lattice_file(),
        name="kagome-stripe-double",
        parms={"L": 2, "a": 1},
    )
    model_descr = model_from_alps_xml(name="spin")
    system = SystemDescriptor(latt_descr, model_descr, parms={"J3": -37.3})
    hamiltonian = system.global_operator("Hamiltonian").simplify().flat()
    target = frozenset(
        (
            "9[0]",
            "10[0]",
        )
    )
    terms = [term for term in hamiltonian.terms if term.acts_over() == target]
    assert len(terms) == 1, f"{len(terms)}!=1."
    assert (
        terms[0].operator[1, 1] == 37.3 / 4.0
    ), f"wrong value found in the  operator matrix element.{terms[0].operator}"
