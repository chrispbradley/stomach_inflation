#!/usr/bin/env python

#> \file
#> \author Chris Bradley
#> \brief This is an example script to solve a finite elasticity problem of stomach inflation OpenCMISS calls in python.
#>
#> \section LICENSE
#>
#> Version: MPL 1.1/GPL 2.0/LGPL 2.1
#>
#> The contents of this file are subject to the Mozilla Public License
#> Version 1.1 (the "License"); you may not use this file except in
#> compliance with the License. You may obtain a copy of the License at
#> http://www.mozilla.org/MPL/
#>
#> Software distributed under the License is distributed on an "AS IS"
#> basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
#> License for the specific language governing rights and limitations
#> under the License.
#>
#> The Original Code is OpenCMISS
#>
#> The Initial Developer of the Original Code is University of Auckland,
#> Auckland, New Zealand and University of Oxford, Oxford, United
#> Kingdom. Portions created by the University of Auckland and University
#> of Oxford are Copyright (C) 2007 by the University of Auckland and
#> the University of Oxford. All Rights Reserved.
#>
#> Contributor(s): Chris Bradley
#>
#> Alternatively, the contents of this file may be used under the terms of
#> either the GNU General Public License Version 2 or later (the "GPL"), or
#> the GNU Lesser General Public License Version 2.1 or later (the "LGPL"),
#> in which case the provisions of the GPL or the LGPL are applicable instead
#> of those above. if you wish to allow use of your version of this file only
#> under the terms of either the GPL or the LGPL, and not to allow others to
#> use your version of this file under the terms of the MPL, indicate your
#> decision by deleting the provisions above and replace them with the notice
#> and other provisions required by the GPL or the LGPL. if you do not delete
#> the provisions above, a recipient may use your version of this file under
#> the terms of any one of the MPL, the GPL or the LGPL.
#>

#> Main script
# Add Python bindings directory to PATH
import sys, os
import numpy as np
import re
import pdb
import math
# Intialise OpenCMISS-Iron
from opencmiss.iron import iron
# Set problem parameters

# Interpolation types
CONSTANT = 0
LINEAR = 1
QUADRATIC = 2
CUBIC = 3

# Constituative law types
MOONEY_RIVLIN_CONSTITUTIVE_LAW = 1
GUCCIONE_CONSTITUTIVE_LAW = 2

# Fibre parameters
ONE_FIBRE_ANGLE = 1
THREE_FIBRE_ANGLES = 2

# Mooney-Rivlin parameters
MOONEY_RIVLIN_1 = 2.0
MOONEY_RIVLIN_2 = 6.0

# Guccione parameters
#GUCCIONE_1 = 5.0
#GUCCIONE_2 = 1.0
#GUCCIONE_3 = 1.0
#GUCCIONE_4 = 1.0
GUCCIONE_1 = 4.0
GUCCIONE_2 = 8.60937380179457
GUCCIONE_3 = 3.66592870916253
GUCCIONE_4 = 25.7671807349653

# Fibre angle(s) in radians
FIBRE_ANGLE=math.pi/4.0
DELTA_ANGLE=math.pi/6.0

#FIBRE_ANGLE_TYPE = ONE_FIBRE_ANGLE
FIBRE_ANGLE_TYPE = THREE_FIBRE_ANGLES

# Material parameters
DENSITY = 9.0e-4

# Constitutive law
#constitutiveLaw = MOONEY_RIVLIN_CONSTITUTIVE_LAW
constitutiveLaw = GUCCIONE_CONSTITUTIVE_LAW

# Interpolation parameters
LINEAR_MESH_COMPONENT = 1 
CUBIC_MESH_COMPONENT = 2

#geometricInterpolation = LINEAR
geometricInterpolation = CUBIC
fibreInterpolation = LINEAR
#displacementInterpolation = LINEAR
displacementInterpolation = CUBIC
hydrostaticPressureInterpolation = CONSTANT
#hydrostaticPressureInterpolation = LINEAR
stressInterpolation = LINEAR
#stressInterpolation = CUBIC
if (displacementInterpolation == CUBIC):
    numberOfGaussXi = 4
else:
    numberOfGaussXi = 3

# Loading parameters
if (constitutiveLaw == MOONEY_RIVLIN_CONSTITUTIVE_LAW):
    pressureSteps = [0.000,0.025,0.050,0.075, \
                     0.100,0.125,0.150,0.175, \
                     0.200,0.225,0.250,0.275, \
                     0.300,0.325,0.350,0.375, \
                     0.400,0.425,0.450,0.475, \
                     0.500,0.525,0.550,0.575, \
                     0.600,0.625,0.650,0.675, \
                     0.700,0.725,0.750,0.775, \
                     0.800,0.825,0.850,0.875, \
                     0.900,0.925,0.950,0.975, \
                     1.000]
elif (constitutiveLaw == GUCCIONE_CONSTITUTIVE_LAW):
    pressureSteps = [0.000,0.025,0.050,0.075, \
                     0.100,0.125,0.150,0.175, \
                     0.200,0.225,0.250,0.275, \
                     0.300,0.325,0.350,0.375, \
                     0.400,0.425,0.450,0.475, \
                     0.500,0.525,0.550,0.575, \
                     0.600,0.625,0.650,0.675, \
                     0.700,0.725,0.750,0.775, \
                     0.800,0.825,0.850,0.875, \
                     0.900,0.925,0.950,0.975, \
                     1.000]
else:
    print("Invalid constitutive law\n")
    exit()              

if (constitutiveLaw == MOONEY_RIVLIN_CONSTITUTIVE_LAW):
    pInit = -MOONEY_RIVLIN_1 # Initial hydrostatic pressure
else:
    pInit = 0.0 # Initial hydrostatic pressure
pReference = pInit # Reference hydrostatic pressure

# Fitting smoothing parameters
tau1 = 0.001
tau2 = 0.001
tau3 = 0.001
kappa11 = 0.00005
kappa12 = 0.00005
kappa22 = 0.00005
kappa13 = 0.00005
kappa23 = 0.00005
kappa33 = 0.00005

# Should not need to change anything below here
CONTEXT_USER_NUMBER = 1
COORDINATE_SYSTEM_USER_NUMBER = 1
REGION_USER_NUMBER = 1
LINEAR_BASIS_USER_NUMBER = 1
QUADRATIC_BASIS_USER_NUMBER = 2
CUBIC_BASIS_USER_NUMBER = 3
MESH_USER_NUMBER = 1
DECOMPOSITION_USER_NUMBER = 1
DECOMPOSER_USER_NUMBER = 1
GEOMETRIC_FIELD_USER_NUMBER = 1
FIBRE_FIELD_USER_NUMBER = 2
ELASTICITY_EQUATIONS_SET_USER_NUMBER = 1
ELASTICITY_EQUATIONS_SET_FIELD_USER_NUMBER = 3
ELASTICITY_DEPENDENT_FIELD_USER_NUMBER = 4
ELASTICITY_MATERIALS_FIELD_USER_NUMBER = 5
ELASTICITY_STRESS_FIELD_USER_NUMBER = 6
ELASTICITY_PROBLEM_USER_NUMBER = 1
FITTING_EQUATIONS_SET_USER_NUMBER = 2
FITTING_EQUATIONS_SET_FIELD_USER_NUMBER = 9
FITTING_DEPENDENT_FIELD_USER_NUMBER = 10
FITTING_MATERIALS_FIELD_USER_NUMBER = 11
FITTING_PROBLEM_USER_NUMBER = 2

numberOfGauss = pow(numberOfGaussXi,3)
 
iron.OutputSetOn("Stomach")
   
context = iron.Context()
context.Create(CONTEXT_USER_NUMBER)

worldRegion = iron.Region()
context.WorldRegionGet(worldRegion)

# Get the number of computational nodes and this computational node number
computationEnvironment = iron.ComputationEnvironment()
context.ComputationEnvironmentGet(computationEnvironment)

worldWorkGroup = iron.WorkGroup()
computationEnvironment.WorldWorkGroupGet(worldWorkGroup)
numberOfComputationalNodes = worldWorkGroup.NumberOfGroupNodesGet()
computationalNodeNumber = worldWorkGroup.GroupNodeNumberGet()

# Set up what types of interpolation to use
if (geometricInterpolation == LINEAR):
    geometricMeshComponent = LINEAR_MESH_COMPONENT
    geometricDerivatives = [iron.GlobalDerivativeConstants.NO_GLOBAL_DERIV]
elif (geometricInterpolation == CUBIC):
    geometricMeshComponent = CUBIC_MESH_COMPONENT
    geometricDerivatives = list(range(iron.GlobalDerivativeConstants.NO_GLOBAL_DERIV, \
                                      iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1_S2_S3+1))
else:
    print("Invalid geometric interpolation\n")
    exit()
if (fibreInterpolation == LINEAR):
    fibreMeshComponent = LINEAR_MESH_COMPONENT
    fibreDerivatives = [iron.GlobalDerivativeConstants.NO_GLOBAL_DERIV]
elif (fibreInterpolation == CUBIC):
    fibreMeshComponent = CUBIC_MESH_COMPONENT
    fibreDerivatives = list(range(iron.GlobalDerivativeConstants.NO_GLOBAL_DERIV, \
                                  iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1_S2_S3+1))
else:
    print("Invalid fibre interpolation\n")
    exit()
if (displacementInterpolation == LINEAR):
    displacementMeshComponent = LINEAR_MESH_COMPONENT
    displacementDerivatives1 = [iron.GlobalDerivativeConstants.NO_GLOBAL_DERIV]
    if (hydrostaticPressureInterpolation == CONSTANT):
          hydrostaticPressureMeshComponent = LINEAR_MESH_COMPONENT
    else:
          print("Invalid hydrostatic pressure interpolation for linear displacement\n")
          exit()          
elif (displacementInterpolation == CUBIC):
    displacementMeshComponent = CUBIC_MESH_COMPONENT
    displacementDerivatives1 = list(range(iron.GlobalDerivativeConstants.NO_GLOBAL_DERIV, \
                                          iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1_S2_S3+1))
    displacementDerivatives2 = [iron.GlobalDerivativeConstants.NO_GLOBAL_DERIV]
    list.extend(displacementDerivatives2,list(range(iron.GlobalDerivativeConstants.GLOBAL_DERIV_S2, \
                                                    iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1_S2_S3+1)))
    if (hydrostaticPressureInterpolation == CONSTANT):
          hydrostaticPressureMeshComponent = LINEAR_MESH_COMPONENT
    elif (hydrostaticPressureInterpolation == LINEAR):
          hydrostaticPressureMeshComponent = LINEAR_MESH_COMPONENT
          hydrostaticPressureDerivatives = [iron.GlobalDerivativeConstants.NO_GLOBAL_DERIV]
    else:
          print("Invalid hydrostatic pressure interpolation for cubic displacement\n")
          exit()          
else:
    print("Invalid displacement interpolation\n")
    exit()
if (stressInterpolation == LINEAR):
    stressMeshComponent = LINEAR_MESH_COMPONENT
    stressDerivatives1 = [iron.GlobalDerivativeConstants.NO_GLOBAL_DERIV]
    stressDerivatives2 = [iron.GlobalDerivativeConstants.NO_GLOBAL_DERIV]
elif (stressInterpolation == CUBIC):
    stressMeshComponent = CUBIC_MESH_COMPONENT
    stressDerivatives1 = list(range(iron.GlobalDerivativeConstants.NO_GLOBAL_DERIV, \
                                    iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1_S2_S3+1))
    stressDerivatives2 = [iron.GlobalDerivativeConstants.NO_GLOBAL_DERIV]
    list.extend(stressDerivatives2,list(range(iron.GlobalDerivativeConstants.GLOBAL_DERIV_S2, \
                                              iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1_S2_S3+1)))
else:
    print("Invalid stress interpolation\n")
    exit()

print("Setting up stomach geometry ...")

# Create a 3D rectangular cartesian coordinate system
coordinateSystem = iron.CoordinateSystem()
coordinateSystem.CreateStart(COORDINATE_SYSTEM_USER_NUMBER,context)
coordinateSystem.DimensionSet(3)
coordinateSystem.CreateFinish()

# Create a region and assign the coordinate system to the region
region = iron.Region()
region.CreateStart(REGION_USER_NUMBER,worldRegion)
region.LabelSet("StomachRegion")
region.CoordinateSystemSet(coordinateSystem)
region.CreateFinish()

# Define linear basis
linearBasis = iron.Basis()
linearBasis.CreateStart(LINEAR_BASIS_USER_NUMBER,context)
linearBasis.TypeSet(iron.BasisTypes.LAGRANGE_HERMITE_TP)
linearBasis.NumberOfXiSet(3)
linearBasis.InterpolationXiSet([iron.BasisInterpolationSpecifications.LINEAR_LAGRANGE]*3)
linearBasis.QuadratureNumberOfGaussXiSet([numberOfGaussXi]*3)
linearBasis.CreateFinish()

# Define quadratic basis
quadraticBasis = iron.Basis()
quadraticBasis.CreateStart(QUADRATIC_BASIS_USER_NUMBER,context)
quadraticBasis.TypeSet(iron.BasisTypes.LAGRANGE_HERMITE_TP)
quadraticBasis.NumberOfXiSet(3)
quadraticBasis.InterpolationXiSet([iron.BasisInterpolationSpecifications.QUADRATIC_LAGRANGE]*3)
quadraticBasis.QuadratureNumberOfGaussXiSet([numberOfGaussXi]*3)
quadraticBasis.CreateFinish()

# Define cubic Hermite basis
cubicBasis = iron.Basis()
cubicBasis.CreateStart(CUBIC_BASIS_USER_NUMBER,context)
cubicBasis.TypeSet(iron.BasisTypes.LAGRANGE_HERMITE_TP)
cubicBasis.NumberOfXiSet(3)
cubicBasis.InterpolationXiSet([iron.BasisInterpolationSpecifications.CUBIC_HERMITE]*3)
cubicBasis.QuadratureNumberOfGaussXiSet([numberOfGaussXi]*3)
cubicBasis.CreateFinish()

# Read data from files
fid= open('stomach_final.ipelem') 
lines=fid.readlines()

numberOfElements=[int(s) for s in lines[3].split() if s.isdigit()][-1]            
elementConnectivity=np.zeros((numberOfElements,9),dtype=np.int32)

for elementIdx in range(1,numberOfElements+1):
    elementConnectivity[elementIdx-1][0]=[int(s) for s in lines[5+8*(elementIdx-1)].split() if s.isdigit()][-1]
    elementConnectivity[elementIdx-1][1:]=[int(s) for s in lines[10+8*(elementIdx-1)].split() if s.isdigit()][1:]

fid.close()

fid = open('stomach_final.ipnode')
lines=fid.readlines()

numberOfNodes=[int(s) for s in lines[3].split() if s.isdigit()][0]
nodalCoordinates=np.zeros((numberOfNodes,25))

for nodeIdx in range(0,numberOfNodes):
     for i in range(1,24):
         nodalCoordinates[nodeIdx][0]=lines[12+(nodeIdx)*26].split()[4]
         nodalCoordinates[nodeIdx][i]=lines[12+(nodeIdx)*26+i].split()[-1]

inletNodes1=[3,4]
list.extend(inletNodes1,list(range(21,27)))
list.extend(inletNodes1,list(range(137,145)))
list.extend(inletNodes1,list(range(193,209)))
inletNodes2=[5,28,46,56,61,62,95,96]
list.extend(inletNodes2,list(range(129,137)))
inletNodes=list(inletNodes1)
list.extend(inletNodes,inletNodes2)
outletNodes1=[1,2]
list.extend(outletNodes1,list(range(15,21)))
list.extend(outletNodes1,list(range(185,193)))
list.extend(outletNodes1,list(range(209,225)))
outletNodes2=[14,32,37,47,68,69,93,94]
list.extend(outletNodes2,list(range(177,185)))
outletNodes=list(outletNodes1)
list.extend(outletNodes,outletNodes2)
stomachSurface=list(range(97,129))
list.extend(stomachSurface,list(range(145,177)))
inletElements=list(range(81,97))
outletElements=list(range(1,9))
list.extend(outletElements,list(range(97,105)))

# Define the mesh in the region
mesh = iron.Mesh()
mesh.CreateStart(MESH_USER_NUMBER,region,3)
mesh.NumberOfComponentsSet(2)
mesh.NumberOfElementsSet(numberOfElements)

# Define nodes for the mesh
nodes = iron.Nodes()
nodes.CreateStart(region,numberOfNodes)
nodes.CreateFinish()

# Define linear mesh elements
linearElements = iron.MeshElements()
linearElements.CreateStart(mesh,LINEAR_MESH_COMPONENT,linearBasis)
for elementIdx in range(1,numberOfElements+1 ):
    linearElements.NodesSet(elementIdx,elementConnectivity[elementIdx-1][1:])
linearElements.CreateFinish()

# Define cubic mesh elements
cubicElements = iron.MeshElements()
cubicElements.CreateStart(mesh,CUBIC_MESH_COMPONENT,cubicBasis)
for elementIdx in range(1,numberOfElements+1 ):
    cubicElements.NodesSet(elementIdx,elementConnectivity[elementIdx-1][1:])
cubicElements.CreateFinish()
mesh.CreateFinish()

# Create a decomposition for the mesh
decomposition = iron.Decomposition()
decomposition.CreateStart(DECOMPOSITION_USER_NUMBER,mesh)
decomposition.CreateFinish()

# Decompose 
decomposer = iron.Decomposer()
decomposer.CreateStart(DECOMPOSER_USER_NUMBER,worldRegion,worldWorkGroup)
decompositionIndex = decomposer.DecompositionAdd(decomposition)
decomposer.CreateFinish()

# Create a field for the geometry
geometricField = iron.Field()
geometricField.CreateStart(GEOMETRIC_FIELD_USER_NUMBER,region)
geometricField.DecompositionSet(decomposition)
geometricField.TypeSet(iron.FieldTypes.GEOMETRIC)
geometricField.VariableLabelSet(iron.FieldVariableTypes.U,"Geometry")
geometricField.ComponentMeshComponentSet(iron.FieldVariableTypes.U,1,geometricMeshComponent)
geometricField.ComponentMeshComponentSet(iron.FieldVariableTypes.U,2,geometricMeshComponent)
geometricField.ComponentMeshComponentSet(iron.FieldVariableTypes.U,3,geometricMeshComponent)
geometricField.ScalingTypeSet(iron.FieldScalingTypes.UNIT)
geometricField.CreateFinish()

# Update the geometric field parameters
for nodeIdx in range(1,numberOfNodes+1):
    nodeDomain = decomposition.NodeDomainGet(nodeIdx,geometricMeshComponent)
    if nodeDomain == computationalNodeNumber:
        for coordinateIdx in range(1,4):
            for derivativeIdx in geometricDerivatives:
                geometricField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,\
                                                        1,derivativeIdx,nodeIdx,coordinateIdx,\
                                                        nodalCoordinates[nodeIdx-1][derivativeIdx+(coordinateIdx-1)*8])
                                                  
geometricField.ParameterSetUpdateStart(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES)
geometricField.ParameterSetUpdateFinish(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES)

# Create a fibre field and attach it to the geometric field
fibreField = iron.Field()
fibreField.CreateStart(FIBRE_FIELD_USER_NUMBER,region)
fibreField.TypeSet(iron.FieldTypes.FIBRE)
fibreField.DecompositionSet(decomposition)
fibreField.GeometricFieldSet(geometricField)
fibreField.VariableLabelSet(iron.FieldVariableTypes.U,"Fibre")
fibreField.ComponentMeshComponentSet(iron.FieldVariableTypes.U,1,fibreMeshComponent)
fibreField.ComponentMeshComponentSet(iron.FieldVariableTypes.U,2,fibreMeshComponent)
fibreField.ComponentMeshComponentSet(iron.FieldVariableTypes.U,3,fibreMeshComponent)
if (FIBRE_ANGLE_TYPE == THREE_FIBRE_ANGLES):
    fibreField.ComponentInterpolationSet(iron.FieldVariableTypes.U,1,iron.FieldInterpolationTypes.GAUSS_POINT_BASED)
    fibreField.ComponentInterpolationSet(iron.FieldVariableTypes.U,2,iron.FieldInterpolationTypes.GAUSS_POINT_BASED)
    fibreField.ComponentInterpolationSet(iron.FieldVariableTypes.U,3,iron.FieldInterpolationTypes.GAUSS_POINT_BASED)
fibreField.ScalingTypeSet(iron.FieldScalingTypes.UNIT)
fibreField.CreateFinish()

# Set the fibre angles wrt to xi 1.
if (FIBRE_ANGLE_TYPE == ONE_FIBRE_ANGLE):
    fibreField.ComponentValuesInitialiseDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,1,FIBRE_ANGLE)
elif (FIBRE_ANGLE_TYPE == THREE_FIBRE_ANGLES):
    for elementIdx in range(1,numberOfElements+1):
        gaussPoint = 0
        for xi3Idx in range (1,numberOfGaussXi+1):
            for xi2Idx in range (1,numberOfGaussXi+1):
                for xi1Idx in range (1,numberOfGaussXi+1):
                    gaussPoint=gaussPoint+1
                    if (xi3Idx == 1):
                        fibreField.ParameterSetUpdateGaussPointDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,\
                                                                  gaussPoint,elementIdx,1,FIBRE_ANGLE-DELTA_ANGLE)
                    elif (xi3Idx == numberOfGaussXi):
                        fibreField.ParameterSetUpdateGaussPointDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,\
                                                                  gaussPoint,elementIdx,1,FIBRE_ANGLE+DELTA_ANGLE)
                    else:
                        fibreField.ParameterSetUpdateGaussPointDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,\
                                                                  gaussPoint,elementIdx,1,FIBRE_ANGLE)
else:
     sys.exit("Invalid fibre angles")

fibreField.ParameterSetUpdateStart(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES)
fibreField.ParameterSetUpdateFinish(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES)

print("Setting up elasticity equations set ...")

# Create the elasticity equations_set
elasticityEquationsSetField = iron.Field()
elasticityEquationsSet = iron.EquationsSet()
if (constitutiveLaw == MOONEY_RIVLIN_CONSTITUTIVE_LAW):
    elasticityEquationsSetSpecification = [iron.EquationsSetClasses.ELASTICITY, \
                                           iron.EquationsSetTypes.FINITE_ELASTICITY, \
                                           iron.EquationsSetSubtypes.MOONEY_RIVLIN]
elif (constitutiveLaw == GUCCIONE_CONSTITUTIVE_LAW):
    elasticityEquationsSetSpecification = [iron.EquationsSetClasses.ELASTICITY, \
                                           iron.EquationsSetTypes.FINITE_ELASTICITY, \
                                           iron.EquationsSetSubtypes.TRANSVERSE_ISOTROPIC_GUCCIONE]
else:
    print('Invalid constitutive law.')
    exit()
    
elasticityEquationsSet.CreateStart(ELASTICITY_EQUATIONS_SET_USER_NUMBER,region,fibreField, \
                         elasticityEquationsSetSpecification,ELASTICITY_EQUATIONS_SET_FIELD_USER_NUMBER, \
                         elasticityEquationsSetField)
elasticityEquationsSet.CreateFinish()

# Create the dependent field
elasticityDependentField = iron.Field()
elasticityEquationsSet.DependentCreateStart(ELASTICITY_DEPENDENT_FIELD_USER_NUMBER,elasticityDependentField)
elasticityDependentField.VariableLabelSet(iron.FieldVariableTypes.U,"ElasticityDependent")
elasticityDependentField.VariableLabelSet(iron.FieldVariableTypes.DELUDELN,"ElasticityTraction")
elasticityDependentField.ComponentMeshComponentSet(iron.FieldVariableTypes.U,1,displacementMeshComponent)
elasticityDependentField.ComponentMeshComponentSet(iron.FieldVariableTypes.DELUDELN,1,displacementMeshComponent)
elasticityDependentField.ComponentMeshComponentSet(iron.FieldVariableTypes.U,2,displacementMeshComponent)
elasticityDependentField.ComponentMeshComponentSet(iron.FieldVariableTypes.DELUDELN,2,displacementMeshComponent)
elasticityDependentField.ComponentMeshComponentSet(iron.FieldVariableTypes.U,3,displacementMeshComponent)
elasticityDependentField.ComponentMeshComponentSet(iron.FieldVariableTypes.DELUDELN,3,displacementMeshComponent)
if (hydrostaticPressureInterpolation == CONSTANT):
    # Set the pressure to be element based
    elasticityDependentField.ComponentInterpolationSet(iron.FieldVariableTypes.U, \
                                                       4,iron.FieldInterpolationTypes.ELEMENT_BASED)
    elasticityDependentField.ComponentInterpolationSet(iron.FieldVariableTypes.DELUDELN, \
                                                       4,iron.FieldInterpolationTypes.ELEMENT_BASED)    
else:
    # Set the pressure to be nodally based
    elasticityDependentField.ComponentInterpolationSet(iron.FieldVariableTypes.U,4,iron.FieldInterpolationTypes.NODE_BASED)
    elasticityDependentField.ComponentInterpolationSet(iron.FieldVariableTypes.DELUDELN,4,iron.FieldInterpolationTypes.NODE_BASED)
elasticityDependentField.ComponentMeshComponentSet(iron.FieldVariableTypes.U,4,hydrostaticPressureMeshComponent)
elasticityDependentField.ComponentMeshComponentSet(iron.FieldVariableTypes.DELUDELN,4,hydrostaticPressureMeshComponent)
elasticityDependentField.ScalingTypeSet(iron.FieldScalingTypes.UNIT)
elasticityEquationsSet.DependentCreateFinish()

# Initialise elasticity dependent field from undeformed geometry and displacement bcs and set hydrostatic pressure
iron.Field.ParametersToFieldParametersComponentCopy( \
    geometricField,iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,1, \
    elasticityDependentField,iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,1)
iron.Field.ParametersToFieldParametersComponentCopy( \
    geometricField,iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,2, \
    elasticityDependentField,iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,2)
iron.Field.ParametersToFieldParametersComponentCopy( \
    geometricField,iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,3, \
    elasticityDependentField,iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,3)
iron.Field.ComponentValuesInitialiseDP( \
    elasticityDependentField,iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,4,pInit)

# Create a field for the stress field. We will use this as the independent field for fitting to save a field copy. 
elasticityStressField = iron.Field()
elasticityStressField.CreateStart(ELASTICITY_STRESS_FIELD_USER_NUMBER,region)
elasticityStressField.TypeSet(iron.FieldTypes.GENERAL)
elasticityStressField.DecompositionSet(decomposition)
elasticityStressField.GeometricFieldSet(geometricField)
elasticityStressField.DependentTypeSet(iron.FieldDependentTypes.DEPENDENT)
elasticityStressField.NumberOfVariablesSet(2)
elasticityStressField.VariableTypesSet([iron.FieldVariableTypes.U,iron.FieldVariableTypes.V])
elasticityStressField.VariableLabelSet(iron.FieldVariableTypes.U,"GaussStress")
elasticityStressField.VariableLabelSet(iron.FieldVariableTypes.V,"GaussWeight")
for variableType in [iron.FieldVariableTypes.U,iron.FieldVariableTypes.V]:
    elasticityStressField.NumberOfComponentsSet(variableType,6)
    for componentIdx in range(1,7):
        elasticityStressField.ComponentMeshComponentSet(variableType,componentIdx,stressMeshComponent)
        elasticityStressField.ComponentInterpolationSet(variableType,componentIdx,iron.FieldInterpolationTypes.GAUSS_POINT_BASED)
elasticityStressField.ScalingTypeSet(iron.FieldScalingTypes.UNIT)
elasticityStressField.CreateFinish()

# Create the derived equations set stress fields
elasticityEquationsSet.DerivedCreateStart(ELASTICITY_STRESS_FIELD_USER_NUMBER,elasticityStressField)
elasticityEquationsSet.DerivedVariableSet(iron.EquationsSetDerivedTensorTypes.CAUCHY_STRESS,iron.FieldVariableTypes.U)
elasticityEquationsSet.DerivedCreateFinish()

# Create the material field
elasticityMaterialsField = iron.Field()
elasticityEquationsSet.MaterialsCreateStart(ELASTICITY_MATERIALS_FIELD_USER_NUMBER,elasticityMaterialsField)
elasticityMaterialsField.VariableLabelSet(iron.FieldVariableTypes.U,"ElasticityMaterial")
elasticityMaterialsField.VariableLabelSet(iron.FieldVariableTypes.V,"ElasticityDensity")
elasticityEquationsSet.MaterialsCreateFinish()

# Set material parameters
if (constitutiveLaw == MOONEY_RIVLIN_CONSTITUTIVE_LAW):
    elasticityMaterialsField.ComponentValuesInitialiseDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES, \
                                                         1,MOONEY_RIVLIN_1)
    elasticityMaterialsField.ComponentValuesInitialiseDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES, \
                                                         2,MOONEY_RIVLIN_2)
elif (constitutiveLaw == GUCCIONE_CONSTITUTIVE_LAW):
    elasticityMaterialsField.ComponentValuesInitialiseDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES, \
                                                         1,GUCCIONE_1)
    elasticityMaterialsField.ComponentValuesInitialiseDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES, \
                                                         2,GUCCIONE_2)
    elasticityMaterialsField.ComponentValuesInitialiseDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES, \
                                                         3,GUCCIONE_3)
    elasticityMaterialsField.ComponentValuesInitialiseDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES, \
                                                         4,GUCCIONE_4)
else:
    print('Invalid constitutive law.')
    exit()
elasticityMaterialsField.ComponentValuesInitialiseDP(iron.FieldVariableTypes.V,iron.FieldParameterSetTypes.VALUES, \
                                                     1,DENSITY)

# Create elasticity equations
elasticityEquations = iron.Equations()
elasticityEquationsSet.EquationsCreateStart(elasticityEquations)
elasticityEquations.SparsityTypeSet(iron.EquationsSparsityTypes.SPARSE)
elasticityEquations.OutputTypeSet(iron.EquationsOutputTypes.NONE)
elasticityEquationsSet.EquationsCreateFinish()

print("Setting up fitting equations set ...")

# Create the fitting equations_set
fittingEquationsSetField = iron.Field()
fittingEquationsSet = iron.EquationsSet()
fittingEquationsSetSpecification = [iron.EquationsSetClasses.FITTING, \
                                    iron.EquationsSetTypes.GAUSS_FITTING_EQUATION, \
                                    iron.EquationsSetSubtypes.GENERALISED_GAUSS_FITTING, \
                                    iron.EquationsSetFittingSmoothingTypes.SOBOLEV_VALUE]
fittingEquationsSet.CreateStart(FITTING_EQUATIONS_SET_USER_NUMBER,region,geometricField, \
                                fittingEquationsSetSpecification,FITTING_EQUATIONS_SET_FIELD_USER_NUMBER, \
                                fittingEquationsSetField)
fittingEquationsSet.CreateFinish()

# Create the fitting dependent field
fittingDependentField = iron.Field()
fittingEquationsSet.DependentCreateStart(FITTING_DEPENDENT_FIELD_USER_NUMBER,fittingDependentField)
fittingDependentField.VariableLabelSet(iron.FieldVariableTypes.U,"FittedStress")
fittingDependentField.NumberOfComponentsSet(iron.FieldVariableTypes.U,6)
for componentIdx in range(1,7):
    fittingDependentField.ComponentMeshComponentSet(iron.FieldVariableTypes.U,componentIdx,stressMeshComponent)
# Finish creating the fitting dependent field
fittingEquationsSet.DependentCreateFinish()

# Create the fitting independent field. Use the previously created elasticity stress field
fittingEquationsSet.IndependentCreateStart(ELASTICITY_STRESS_FIELD_USER_NUMBER,elasticityStressField)
# Finish creating the fitting independent field
fittingEquationsSet.IndependentCreateFinish()

# Initialise Gauss point weight field to 1.0
for componentIdx in range(1,7):
    elasticityStressField.ComponentValuesInitialiseDP(iron.FieldVariableTypes.V, \
                                                      iron.FieldParameterSetTypes.VALUES,componentIdx,1.0)

# Create material field (Sobolev parameters)
fittingMaterialField = iron.Field()
fittingEquationsSet.MaterialsCreateStart(FITTING_MATERIALS_FIELD_USER_NUMBER,fittingMaterialField)
fittingMaterialField.VariableLabelSet(iron.FieldVariableTypes.U,"SmoothingParameters")
fittingEquationsSet.MaterialsCreateFinish()
# Set tau and kappa - Sobolev smoothing parameters
# Set tau's and kappa's - Sobolev smoothing parameters
fittingMaterialField.ComponentValuesInitialiseDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,1,tau1)
fittingMaterialField.ComponentValuesInitialiseDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,2,kappa11)
fittingMaterialField.ComponentValuesInitialiseDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,3,tau2)
fittingMaterialField.ComponentValuesInitialiseDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,4,kappa22)
fittingMaterialField.ComponentValuesInitialiseDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,5,kappa12)
fittingMaterialField.ComponentValuesInitialiseDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,6,tau3)
fittingMaterialField.ComponentValuesInitialiseDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,7,kappa33)
fittingMaterialField.ComponentValuesInitialiseDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,8,kappa13)
fittingMaterialField.ComponentValuesInitialiseDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,9,kappa23)

# Create the fitting equations
fittingEquations = iron.Equations()
fittingEquationsSet.EquationsCreateStart(fittingEquations)
# Set the fitting equations sparsity type
fittingEquations.SparsityTypeSet(iron.EquationsSparsityTypes.SPARSE)
# Set the fitting equations output type to none
fittingEquations.OutputTypeSet(iron.EquationsOutputTypes.NONE)
# Finish creating the fitting equations
fittingEquationsSet.EquationsCreateFinish()


print("Setting up fitting problem...")

# Create fitting problem
fittingProblemSpecification = [iron.ProblemClasses.FITTING, \
                               iron.ProblemTypes.FITTING, \
                               iron.ProblemSubtypes.STATIC_LINEAR_FITTING]
fittingProblem = iron.Problem()
fittingProblem.CreateStart(FITTING_PROBLEM_USER_NUMBER,context,fittingProblemSpecification)
fittingProblem.CreateFinish()

# Create control loops
fittingProblem.ControlLoopCreateStart()
fittingProblem.ControlLoopCreateFinish()

# Create problem solver
fittingSolver = iron.Solver()
fittingProblem.SolversCreateStart()
fittingProblem.SolverGet([iron.ControlLoopIdentifiers.NODE],1,fittingSolver)
fittingSolver.OutputTypeSet(iron.SolverOutputTypes.PROGRESS)
fittingSolver.LinearTypeSet(iron.LinearSolverTypes.DIRECT)
fittingProblem.SolversCreateFinish()

# Create fitting solver equations and add fitting equations set to solver equations
fittingSolverEquations = iron.SolverEquations()
fittingProblem.SolverEquationsCreateStart()
# Get the solver equations
fittingSolver.SolverEquationsGet(fittingSolverEquations)
fittingSolverEquations.SparsityTypeSet(iron.SolverEquationsSparsityTypes.SPARSE)
fittingEquationsSetIndex = fittingSolverEquations.EquationsSetAdd(fittingEquationsSet)
fittingProblem.SolverEquationsCreateFinish()

# Prescribe boundary conditions for the fitting problem
fittingBoundaryConditions = iron.BoundaryConditions()
fittingSolverEquations.BoundaryConditionsCreateStart(fittingBoundaryConditions)
# No stress at the inlet
for nodeIdx in range(0,len(inletNodes1)):
    nodeDomain = decomposition.NodeDomainGet(inletNodes1[nodeIdx],stressMeshComponent)
    if nodeDomain == computationalNodeNumber:
        for componentIdx in range(1,7):
            for derivativeIdx in stressDerivatives1:
                fittingBoundaryConditions.AddNode(fittingDependentField,iron.FieldVariableTypes.U, \
                                                  1,derivativeIdx,inletNodes1[nodeIdx],\
                                                  componentIdx,iron.BoundaryConditionsTypes.FIXED,0.0) #no stress
for nodeIdx in range(0,len(inletNodes2)):
    nodeDomain = decomposition.NodeDomainGet(inletNodes2[nodeIdx],stressMeshComponent)
    if nodeDomain == computationalNodeNumber:
        for componentIdx in range(1,7):
            for derivativeIdx in stressDerivatives2:
                fittingBoundaryConditions.AddNode(fittingDependentField,iron.FieldVariableTypes.U, \
                                                  1,derivativeIdx,inletNodes2[nodeIdx],\
                                                  componentIdx,iron.BoundaryConditionsTypes.FIXED,0.0) #no stress
# No stress at the outlet
for nodeIdx in range(0,len(outletNodes1)):
    nodeDomain = decomposition.NodeDomainGet(outletNodes1[nodeIdx],stressMeshComponent)
    if nodeDomain == computationalNodeNumber:
        for componentIdx in range(1,7):
            for derivativeIdx in stressDerivatives1:
                fittingBoundaryConditions.AddNode(fittingDependentField,iron.FieldVariableTypes.U, \
                                                  1,derivativeIdx,outletNodes1[nodeIdx],\
                                                  componentIdx,iron.BoundaryConditionsTypes.FIXED,0.0) #no stress
for nodeIdx in range(0,len(outletNodes2)):
    nodeDomain = decomposition.NodeDomainGet(outletNodes2[nodeIdx],stressMeshComponent)
    if nodeDomain == computationalNodeNumber:
        for componentIdx in range(1,7):
            for derivativeIdx in stressDerivatives2:
                fittingBoundaryConditions.AddNode(fittingDependentField,iron.FieldVariableTypes.U, \
                                                  1,derivativeIdx,outletNodes2[nodeIdx],\
                                                  componentIdx,iron.BoundaryConditionsTypes.FIXED,0.0) #no stress
fittingSolverEquations.BoundaryConditionsCreateFinish()

numberOfLoadIncrements = [1]*len(pressureSteps)

for stepIdx, pressure in enumerate(pressureSteps):

    print("Setting up elasticity problem for step %d..." % stepIdx)

    # Define the elasticity problem
    elasticityProblem = iron.Problem()
    elasticityProblemSpecification = [iron.ProblemClasses.ELASTICITY, \
                                      iron.ProblemTypes.FINITE_ELASTICITY, \
                                      iron.ProblemSubtypes.STATIC_FINITE_ELASTICITY]
    elasticityProblem.CreateStart(ELASTICITY_PROBLEM_USER_NUMBER,context,elasticityProblemSpecification)
    elasticityProblem.CreateFinish()
    
    # Create the elasticity problem control loop
    elasticityProblem.ControlLoopCreateStart()
    elasticityControlLoop = iron.ControlLoop()
    elasticityProblem.ControlLoopGet([iron.ControlLoopIdentifiers.NODE],elasticityControlLoop)
    elasticityControlLoop.MaximumIterationsSet(numberOfLoadIncrements[stepIdx])
    elasticityProblem.ControlLoopCreateFinish()
    
    # Create elasticity problem solvers
    elasticityNonLinearSolver = iron.Solver()
    elasticityLinearSolver = iron.Solver()
    elasticityProblem.SolversCreateStart()
    elasticityProblem.SolverGet([iron.ControlLoopIdentifiers.NODE],1,elasticityNonLinearSolver)
    elasticityNonLinearSolver.OutputTypeSet(iron.SolverOutputTypes.MONITOR)
    #elasticityNonLinearSolver.OutputTypeSet(iron.SolverOutputTypes.PROGRESS)
    #elasticityNonLinearSolver.OutputTypeSet(iron.SolverOutputTypes.MATRIX)
    if (constitutiveLaw == MOONEY_RIVLIN_CONSTITUTIVE_LAW):
        elasticityNonLinearSolver.NewtonJacobianCalculationTypeSet(iron.JacobianCalculationTypes.FD)
    elif (constitutiveLaw == GUCCIONE_CONSTITUTIVE_LAW):
        elasticityNonLinearSolver.NewtonJacobianCalculationTypeSet(iron.JacobianCalculationTypes.EQUATIONS)
    else:
        print("Invalid constitutive law\n")
        exit()              
    elasticityNonLinearSolver.NewtonAbsoluteToleranceSet(1e-8)
    elasticityNonLinearSolver.NewtonSolutionToleranceSet(1e-8)
    elasticityNonLinearSolver.NewtonRelativeToleranceSet(1e-8)
    elasticityNonLinearSolver.NewtonLinearSolverGet(elasticityLinearSolver)
    elasticityLinearSolver.LinearTypeSet(iron.LinearSolverTypes.DIRECT)
    elasticityProblem.SolversCreateFinish()

    # Create elasticity solver equations and add elasticity equations set to solver equations
    elasticitySolverEquations = iron.SolverEquations()
    elasticityProblem.SolverEquationsCreateStart()
    elasticityNonLinearSolver.SolverEquationsGet(elasticitySolverEquations)
    elasticitySolverEquations.SparsityTypeSet(iron.SolverEquationsSparsityTypes.SPARSE)
    elasticityEquationsSetIndex = elasticitySolverEquations.EquationsSetAdd(elasticityEquationsSet)
    elasticityProblem.SolverEquationsCreateFinish()
    
    # Prescribe boundary conditions for the elasticity problem
    elasticityBoundaryConditions = iron.BoundaryConditions()
    elasticitySolverEquations.BoundaryConditionsCreateStart(elasticityBoundaryConditions)
    
    if (displacementInterpolation == CUBIC):
        # No displacement at the inlet
        for nodeIdx in range(0,len(inletNodes1)):
            nodeDomain = decomposition.NodeDomainGet(inletNodes1[nodeIdx],displacementMeshComponent)
            if nodeDomain == computationalNodeNumber:
                for derivativeIdx in displacementDerivatives1:
                    elasticityBoundaryConditions.AddNode(elasticityDependentField,iron.FieldVariableTypes.U, \
                                                         1,derivativeIdx,inletNodes1[nodeIdx],1,\
                                                         iron.BoundaryConditionsTypes.FIXED,0.0) #no x1 displacement
                    elasticityBoundaryConditions.AddNode(elasticityDependentField,iron.FieldVariableTypes.U,\
                                                         1,derivativeIdx,inletNodes1[nodeIdx],2,\
                                                         iron.BoundaryConditionsTypes.FIXED,0.0) #no x2 displacement
                    elasticityBoundaryConditions.AddNode(elasticityDependentField,iron.FieldVariableTypes.U,\
                                                         1,derivativeIdx,inletNodes1[nodeIdx],3,\
                                                         iron.BoundaryConditionsTypes.FIXED,0.0) #no x3 displacement
        for nodeIdx in range(0,len(inletNodes2)):
            nodeDomain = decomposition.NodeDomainGet(inletNodes2[nodeIdx],displacementMeshComponent)
            if nodeDomain == computationalNodeNumber:
                for derivativeIdx in displacementDerivatives2:
                    elasticityBoundaryConditions.AddNode(elasticityDependentField,iron.FieldVariableTypes.U,\
                                                         1,derivativeIdx,inletNodes2[nodeIdx],1,\
                                                         iron.BoundaryConditionsTypes.FIXED,0.0) #no x1 displacement
                    elasticityBoundaryConditions.AddNode(elasticityDependentField,iron.FieldVariableTypes.U,\
                                                         1,derivativeIdx,inletNodes2[nodeIdx],2,\
                                                         iron.BoundaryConditionsTypes.FIXED,0.0) #no x2 displacement
                    elasticityBoundaryConditions.AddNode(elasticityDependentField,iron.FieldVariableTypes.U,\
                                                         1,derivativeIdx,inletNodes2[nodeIdx],3,\
                                                         iron.BoundaryConditionsTypes.FIXED,0.0) #no x3 displacement
        # No displacement at the outlet
        for nodeIdx in range(0,len(outletNodes1)):
            nodeDomain = decomposition.NodeDomainGet(outletNodes1[nodeIdx],displacementMeshComponent)
            if nodeDomain == computationalNodeNumber:
                for derivativeIdx in displacementDerivatives1:
                    elasticityBoundaryConditions.AddNode(elasticityDependentField,iron.FieldVariableTypes.U,\
                                                         1,derivativeIdx,outletNodes1[nodeIdx],1,\
                                                         iron.BoundaryConditionsTypes.FIXED,0.0)
                    elasticityBoundaryConditions.AddNode(elasticityDependentField,iron.FieldVariableTypes.U,\
                                                         1,derivativeIdx,outletNodes1[nodeIdx],2,\
                                                         iron.BoundaryConditionsTypes.FIXED,0.0)
                    elasticityBoundaryConditions.AddNode(elasticityDependentField,iron.FieldVariableTypes.U,\
                                                         1,derivativeIdx,outletNodes1[nodeIdx],3,\
                                                         iron.BoundaryConditionsTypes.FIXED,0.0)
        for nodeIdx in range(0,len(outletNodes2)):
            nodeDomain = decomposition.NodeDomainGet(outletNodes2[nodeIdx],displacementMeshComponent)
            if nodeDomain == computationalNodeNumber:
                for derivativeIdx in displacementDerivatives2:
                    elasticityBoundaryConditions.AddNode(elasticityDependentField,iron.FieldVariableTypes.U,\
                                                         1,derivativeIdx,outletNodes2[nodeIdx],1,\
                                                         iron.BoundaryConditionsTypes.FIXED,0.0)
                    elasticityBoundaryConditions.AddNode(elasticityDependentField,iron.FieldVariableTypes.U,\
                                                         1,derivativeIdx,outletNodes2[nodeIdx],2,\
                                                         iron.BoundaryConditionsTypes.FIXED,0.0)
                    elasticityBoundaryConditions.AddNode(elasticityDependentField,iron.FieldVariableTypes.U,\
                                                         1,derivativeIdx,outletNodes2[nodeIdx],3,\
                                                         iron.BoundaryConditionsTypes.FIXED,0.0)
    else:
        # No displacement at the inlet
        for nodeIdx in range(0,len(inletNodes)):
            nodeDomain = decomposition.NodeDomainGet(inletNodes[nodeIdx],displacementMeshComponent)
            if nodeDomain == computationalNodeNumber:
                for derivativeIdx in displacementDerivatives1:
                    elasticityBoundaryConditions.AddNode(elasticityDependentField,iron.FieldVariableTypes.U,\
                                                         1,derivativeIdx,inletNodes[nodeIdx],1,\
                                                         iron.BoundaryConditionsTypes.FIXED,0.0) #no x1 displacement
                    elasticityBoundaryConditions.AddNode(elasticityDependentField,iron.FieldVariableTypes.U,\
                                                         1,derivativeIdx,inletNodes[nodeIdx],2,\
                                                         iron.BoundaryConditionsTypes.FIXED,0.0) #no x2 displacement
                    elasticityBoundaryConditions.AddNode(elasticityDependentField,iron.FieldVariableTypes.U,\
                                                         1,derivativeIdx,inletNodes[nodeIdx],3,\
                                                         iron.BoundaryConditionsTypes.FIXED,0.0) #no x3 displacement
                    
        # No displacement at the outlet
        for nodeIdx in range(0,len(outletNodes)):
            nodeDomain = decomposition.NodeDomainGet(outletNodes[nodeIdx],displacementMeshComponent)
            if nodeDomain == computationalNodeNumber:
                for derivativeIdx in displacementDerivatives1:
                    elasticityBoundaryConditions.AddNode(elasticityDependentField,iron.FieldVariableTypes.U,\
                                                         1,derivativeIdx,outletNodes[nodeIdx],1,\
                                                         iron.BoundaryConditionsTypes.FIXED,0.0)
                    elasticityBoundaryConditions.AddNode(elasticityDependentField,iron.FieldVariableTypes.U,\
                                                         1,derivativeIdx,outletNodes[nodeIdx],2,\
                                                         iron.BoundaryConditionsTypes.FIXED,0.0)
                    elasticityBoundaryConditions.AddNode(elasticityDependentField,iron.FieldVariableTypes.U,\
                                                         1,derivativeIdx,outletNodes[nodeIdx],3,\
                                                         iron.BoundaryConditionsTypes.FIXED,0.0)
                    
    # Apply incremented cavity pressure on the stomach surface:
    for nodeIdx in range(0,len(stomachSurface)):
        nodeDomain = decomposition.NodeDomainGet(stomachSurface[nodeIdx],displacementMeshComponent)
        if nodeDomain == computationalNodeNumber:
            # xi_3 is the transmural direction
            xiDirection = 3
            # For pressure/force boundary conditions, the DELUDELN field variable is constrained rather than the U field variable
            elasticityBoundaryConditions.SetNode(elasticityDependentField,iron.FieldVariableTypes.DELUDELN, \
                                                 1,iron.GlobalDerivativeConstants.NO_GLOBAL_DERIV,\
                                                 stomachSurface[nodeIdx],xiDirection, \
                                                 iron.BoundaryConditionsTypes.PRESSURE_INCREMENTED,pressureSteps[stepIdx])

    if (hydrostaticPressureInterpolation == CONSTANT):
        # Fix reference hydrostatic pressure in the inlet elements
        for elementIdx in range(0,len(inletElements)):
            elementDomain = decomposition.ElementDomainGet(inletElements[elementIdx])
            if elementDomain == computationalNodeNumber:
                elasticityBoundaryConditions.SetElement(elasticityDependentField,iron.FieldVariableTypes.U,\
                                                        inletElements[elementIdx],4,\
                                                        iron.BoundaryConditionsTypes.FIXED,pReference) #reference hydrostatic pressure
        # Fix reference hydrostatic pressure in the outlet elements
        for elementIdx in range(0,len(outletElements)):
            elementDomain = decomposition.ElementDomainGet(outletElements[elementIdx])
            if elementDomain == computationalNodeNumber:
                elasticityBoundaryConditions.SetElement(elasticityDependentField,iron.FieldVariableTypes.U,\
                                                        outletElements[elementIdx],4,\
                                                        iron.BoundaryConditionsTypes.FIXED,pReference) #reference hydrostatic pressure
    else:
        # Fix reference hydrostatic pressure at the inlet nodes
        for nodeIdx in range(0,len(inletNodes)):
            nodeDomain = decomposition.NodeDomainGet(inletNodes[nodeIdx],hydrostaticPressureMeshComponent)
            if nodeDomain == computationalNodeNumber:
                elasticityBoundaryConditions.SetNode(elasticityDependentField,iron.FieldVariableTypes.U,\
                                                     1,iron.GlobalDerivativeConstants.NO_GLOBAL_DERIV,\
                                                     inletNodes[nodeIdx],4,iron.BoundaryConditionsTypes.FIXED,pReference) #reference hydrostatic pressure
        # Fix reference hydrostatic pressure at the outlet nodes
        for nodeIdx in range(0,len(outletNodes)):
            nodeDomain = decomposition.NodeDomainGet(outletNodes[nodeIdx],hydrostaticPressureMeshComponent)
            if nodeDomain == computationalNodeNumber:
                elasticityBoundaryConditions.SetNode(elasticityDependentField,iron.FieldVariableTypes.U,\
                                                     1,iron.GlobalDerivativeConstants.NO_GLOBAL_DERIV,\
                                                     outletNodes[nodeIdx],4,iron.BoundaryConditionsTypes.FIXED,pReference) #reference hydrostatic pressure
                
    elasticitySolverEquations.BoundaryConditionsCreateFinish()
    
    # Export results
    fields = iron.Fields()
    fields.CreateRegion(region)
    fields.NodesExport("StomachOriginal","FORTRAN")
    fields.ElementsExport("StomachOriginal","FORTRAN")
    fields.Finalise()

    print("Solving elasticity problem for step %d..." % stepIdx)

    # Solve the elasticity problem
    elasticityProblem.Solve()
    
    print("Solving fitting problem for step %d..." % stepIdx)
    
    # Calculate the stress field
    #elasticityEquationsSet.DerivedVariableCalculate(iron.EquationsSetDerivedTensorTypes.CAUCHY_STRESS)
    
    # Solve the fitting problem
    fittingProblem.Solve()
    
    # Export results
    filename = "Stomach_%d" % stepIdx
    fields = iron.Fields()
    fields.CreateRegion(region)
    fields.NodesExport(filename,"FORTRAN")
    fields.ElementsExport(filename,"FORTRAN")
    fields.Finalise()

    elasticitySolverEquations.Finalise()
    elasticityProblem.Finalise()
