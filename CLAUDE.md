## Introduction
We need to implement the rigid body simulator described here: `docs\rigid_body_simulator\rigid_body_simulator.tex`.
The simulator needs to be in python and will be mainly used to develop our drone control system.

## Coordinate system
I think we will have three coordinate systems:
- World coordinate system: non-inertial. Fixed to the world.
- Body fixed coordinate system: Origin is fixed somewhere on the drone. 
- Gravity coordinate system: Origin is fixed in the CoG of the drone.

The position of the CoG is fixed and described in the body fixed coordinate system.

Essentially I want to have an interface where I can easily "apply" a force or torque on the rigid body where the force can be described in any coordinate system. 
Under the hood it should do the required transformations.

For instance imagine a `RigidBody` class


```python
class CoordinateSystem(StrEnum):
    BODY = 'BODY'
    WORLD = 'WORLD

@dataclass
class Vector():
    components: np.ndarray #3 element vector describing magnitude and direction of the vector
    coordinate_system: CoordinateSystem #Enum describing the coordinate system the above vector is described in

@dataclass
class Force():
    application_point: Vector
    force: Vector

class RigidBody():
    def apply_forces(self, forces:list[Force]):
        pass
    
    def get_acceleration(self, position:Vector) -> Vector:
        # Returns vector in same coordinate system as input
        pass
```

So basically we can define the force in any coordinate system and its application point as well. Internally the sim will translate everything to the correct coordinate system to compute what it needs.
Another feature we need is that we should be able to retrieve the acceleration 

## Typical simulation loop

```python
control_system = ControlSystem()
propellors = Propellors()
cog_vector = Vector(
    components=[0,0.33,0.1], 
    coordinate_system=CoordinateSystem.BODY
)
position_imu_sensor = Vector(
    omponents=[0,0.55,0.15], 
    coordinate_system=CoordinateSystem.BODY
)
drone = RigidBody(I, mass)
F_g = Force(
    application_point=cog_vector, 
    force=Vector(
        components=[0,0,-9.81*mass], 
        coordinate_system=CoordinateSystem.BODY
    ) 
)
solver = RK4()
solver.integratable( #Some sort of list where we register what "things" too integrate the state each step? I guess this means RigidBody inherits from same baseclase with gives defined method on how too extract and set the state
    drone
)
setpoints = ... #SOme input description 
for i in range(N_steps):
    propellor_setpoints = control_system.get_control_points(setpoints, omega, acceleration)
    propellor_forces = propellors.set_speed(propellor_setpoints)
    drone.apply_forces(propellor_forces + [F_g])
    step() #Not sure here we, need to couple the things we want too integrate with a solver
    acceleration = drone.get_acceleration(position_imu_sensor)
    omege = drone.get_rotation_speed(position_imu_sensor)

```

You should not develop the actual simulation loop. Just the foundation rigid body simulator.

## Other stuff
Put all code here: 