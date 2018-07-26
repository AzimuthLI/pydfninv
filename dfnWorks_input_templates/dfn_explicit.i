# Jan 13, 2014
# Nataliia Makedonska, Satish Karra, LANL
#================================================

SIMULATION
  SIMULATION_TYPE SUBSURFACE
  PROCESS_MODELS
    SUBSURFACE_FLOW flow
      MODE RICHARDS
    /
  /
END
SUBSURFACE

DFN

#=========================== discretization ===================================
GRID
  TYPE unstructured_explicit full_mesh_vol_area.uge
  GRAVITY 0.d0 0.d0 0.d0
END


#=========================== fluid properties =================================
FLUID_PROPERTY
  DIFFUSION_COEFFICIENT 1.d-9
END

DATASET Permeability
  FILENAME dfn_properties.h5
END

#=========================== material properties ==============================
MATERIAL_PROPERTY soil1
  ID 1
  POROSITY 0.25d0
  TORTUOSITY 0.5d0
  CHARACTERISTIC_CURVES default
  PERMEABILITY
    DATASET Permeability
  /
END


#=========================== characteristic curves ============================
CHARACTERISTIC_CURVES default
  SATURATION_FUNCTION VAN_GENUCHTEN
    M 0.5d0
    ALPHA  1.d-4
    LIQUID_RESIDUAL_SATURATION 0.1d0
    MAX_CAPILLARY_PRESSURE 1.d8
  /
  PERMEABILITY_FUNCTION MUALEM_VG_LIQ
    M 0.5d0
    LIQUID_RESIDUAL_SATURATION 0.1d0
  /
END

#=========================== output options ===================================
OUTPUT
  TIMES s 0.01 0.05 0.1 0.2 0.5 1

  SNAPSHOT_FILE
    FORMAT VTK MULTIPLE_FILES TIMES_PER_FILE 10
    NO_PRINT_INITIAL
    PERIODIC TIME 10 s
    VARIABLES
      LIQUID_PRESSURE
    /
  /
  PRINT_PRIMAL_GRID
  FORMAT VTK
  MASS_FLOWRATE
  MASS_BALANCE
  VARIABLES
    LIQUID_PRESSURE
    PERMEABILITY
  /
END

#=========================== times ============================================
TIME
  INITIAL_TIMESTEP_SIZE  1.d0 s
  FINAL_TIME 100.d0 s
  MAXIMUM_TIMESTEP_SIZE 10.d0 s
#  STEADY_STATE
END

# REFERENCE_PRESSURE 1500000.

#=========================== regions ==========================================
REGION All
  COORDINATES
    -1.d20 -1.d20 -1.d20
    1.d20 1.d20 1.d20
  /
END

REGION inflow
  FILE $inflow_region
END

REGION outflow
  FILE $outflow_region
END

#=========================== flow conditions ==================================
FLOW_CONDITION initial
  TYPE
     PRESSURE dirichlet
  /
  PRESSURE 1.01325d6
END


FLOW_CONDITION outflow
  TYPE
     PRESSURE dirichlet
  /
  PRESSURE $outflow_pressure
END

FLOW_CONDITION inflow
  TYPE
    PRESSURE dirichlet
  /
  PRESSURE $inflow_pressure
END

#=========================== condition couplers ===============================
# initial condition
INITIAL_CONDITION
  FLOW_CONDITION initial
  REGION All
END


BOUNDARY_CONDITION INFLOW
  FLOW_CONDITION inflow
  REGION inflow
END

BOUNDARY_CONDITION OUTFLOW
  FLOW_CONDITION outflow
  REGION outflow
END

#=========================== stratigraphy couplers ============================
STRATA
  REGION All
  MATERIAL soil1
END

END_SUBSURFACE