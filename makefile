run:
	python Common_Routines.py
	python Filter_Routines.py
	python fixAngleEdge.py
	python MPAS_O_Mode_Init.py
	python MPAS_O_Shared.py
	python MPAS_O_Operator_Testing.py
	python MPAS_O_Mode_Forward.py

clean:	
	rm -r __pycache__/
