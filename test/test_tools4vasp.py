from io import StringIO
from tools4vasp.vaspcheck import _get_elements_from_outcar


file_content = """
asdfasdf
POTCAR: PAW_PBE Au 08Apr2002
POTCAR: PAW_PBE N 08Apr2002
POTCAR: PAW_PBE H 08Apr2002
POTCAR: PAW_PBE C 08Apr2002
asdfasdf
POTCAR: PAW_PBE Au 08Apr2002
asdfasdf
POTCAR: PAW_PBE N 08Apr2002
asdfasdf
POTCAR: PAW_PBE H 08Apr2002
asdfasdf
POTCAR: PAW_PBE C 08Apr2002
asdfasdf
POSCAR: Au N H C
asdfasdf
POSCAR = Au N H C
asdfasdf
"""

file_content_bad = file_content.replace("POSCAR: Au N H C", "POSCAR: Au H N C")
file_content_bad2 = file_content.replace("POTCAR: PAW_PBE N 08Apr2002", "POTCAR: PAW_PBE M 08Apr2002")
file_content_underscore = file_content.replace("Au", "Au_s")

def test_get_elements_from_outcar():
        elements_poscar, elements_potcar = _get_elements_from_outcar(StringIO(file_content))
        poscar = potcar = ["Au", "N", "H", "C"]
        assert elements_poscar == poscar == elements_potcar == potcar

        elements_poscar, elements_potcar = _get_elements_from_outcar(StringIO(file_content_bad))
        assert elements_poscar != poscar
        assert elements_potcar == potcar

        elements_poscar, elements_potcar = _get_elements_from_outcar(StringIO(file_content_bad2))
        assert elements_poscar == poscar
        assert elements_potcar != potcar

        elements_poscar, elements_potcar = _get_elements_from_outcar(StringIO(file_content_underscore))
        assert elements_poscar == poscar == elements_potcar == potcar
