# Problem 2

The drag force on a skydiver is of the form

$$ F_{Drag} = -kv^2 $$

where $k$ = 0.7kgm$^{-1}$s$^2$ with the parachute closed and k = 30kgm$^{-1}$s$^2$ with the parachute open. The skydiver performs a series of 3 jumps from an altitude of 3000m. To perform the jump safely, the speed of the skydiver must be less than 10m/s when they land.

Use the <samp>scipy.integrate.odeint</samp> function to solve the diﬀerential equation describing the sky-diver’s motion. See the Session 4 Advanced Computing Worksheet for help using this function. For each jump listed below plot the altitude and velocity of the skydiver against time and calculate the total time taken for the jump.

a) The parachute is open for the whole jump

b) The jump is performed using a static line 1000 m in length. Therefore the parachute opens when the skydiver is 1000 m below the plane

c) The skydiver deploys their parachute to minimise the total time taken for the jump, whilst still landing safely