import numpy as np 
class Boid(): 
    def __init__(self, x, y, minX, minY, maxX, maxY): 
        # Initial conditions 
        self.position = np.array([x, y]) 

        # Physical constraints 
        self.maxSwimSpeed = 20 # Maximum speed 
        self.maxDelta = 10 # Maximum acceleration 
        self.perceptionDistance = 30 # Maximum distance to see other boids 
        self.avoidanceDistance = 6 # Minimum spacing for avoidance behaviour 

        # Specify our behavioural weights 
        self.aWeight = 0.5 # Alignment, should be 0 ≤ aWeight ≤ 1
        self.cWeight = 1 # Cohesion, should be 0 ≤ cWeight ≤ maxDelta
        self.sWeight = 1 # Avoidance (separation), should be 0 ≤ sWeight ≤ maxDelta
 
        # Initial velocity and acceleration 
        self.velocity = (np.random.rand(2) - 0.5) * self.maxSwimSpeed 
        self.delta = (np.random.rand(2) - 0.5) * self.maxDelta 

        # Study area boundaries
        self.minX = minX
        self.minY = minY
        self.maxX = maxX
        self.maxY = maxY
        
        #Boid's name
        self.boidID = np.random.random()

    def constrainSpeed(self): 
        currentSpeed = np.linalg.norm(self.velocity) 
        if currentSpeed > self.maxSwimSpeed: 
            self.velocity = self.velocity * self.maxSwimSpeed / currentSpeed 

    def constrainDelta(self): 
        currentDelta = np.linalg.norm(self.delta) 
        if currentDelta > self.maxDelta:
             self.delta = self.delta * self.maxDelta / currentDelta

    def outOfBounds(self): 
        nextPosition = self.delta + self.velocity + self.position 
        if nextPosition[0] > self.maxX or nextPosition[0] < self.minX:
            self.velocity[0] *= -1 
            self.delta[0] *= -1
        if nextPosition[1] > self.maxY or nextPosition[1] < self.minY:
            self.velocity[1] *= -1 
            self.delta[1] *= -1

    def update(self): 
        self.constrainSpeed() 
        self.constrainDelta() 
        self.outOfBounds()
        self.position += self.velocity 
        self.velocity += self.delta 
        self.delta = np.zeros(2)

    def align(self, boids): 
        # We need a place to store our the acceleration we come up with 
        # as well as the total number of fish we are averaging across 
        dDelta = np.zeros(2) 
        totalNearby = 0
 
        # Take the total list of fish available to be nearby and calculate 
        # the distance to our fish 
        for boid in boids: 
            difference = boid.position - self.position 
            distance = np.linalg.norm(difference) 

            # If the fish is visible to our fish, add it's velocity to our 
            # target velocity, count the number of nearby fish 
            if distance > 0 and distance < self.perceptionDistance: 
                dDelta += boid.velocity 
                totalNearby += 1 

        # Divide the total velocity of each nearby fish by the number of 
        # nearby fish to get the average velocity, this is our target 
        # velocity. 
        if totalNearby > 0: 
            dDelta /= totalNearby 

            # The acceleration necessary to reach our target velocity in time 
            # step t is simply the target velocity minus our current velocity 
            dDelta -= self.velocity 

        return dDelta

    def cohese(self, boids): 
        # Create a place to store our target acceleration and total visible fish 
        dDelta = np.zeros(2) 
        totalNearby = 0 

        # Check whether any fish are visible to our current fish. Sum their position 
        # if they are visible 
        for boid in boids: 
            difference = boid.position - self.position 
            distance = np.linalg.norm(difference) 
            if distance > 0 and distance < self.perceptionDistance: 
                dDelta += boid.position 
                totalNearby += 1 

        # If any fish were visible, average their positions to find the target 
        # position. 
        if totalNearby > 0: 
            dDelta /= totalNearby 

            # Calculate the velocity necessary to reach the target position. 
            dDelta -= self.position 

            # Normalize the velocity to generate just the direction to accelerate in to 
            # reach the target position 

            magDelta = np.linalg.norm(dDelta) 
            if magDelta > 0: # Make sure we are not at the center of the group (avoid /0) 
                dDelta /= magDelta 
            else: # Do nothing if we are already at the middle. 
                dDelta = np.zeros(2) 

        return dDelta

    def avoid(self, boids): 
        # Create a place to store our target acceleration and the total fish 
        # we want to avoid colliding with. 
        dDelta = np.zeros(2) 
        totalNearby = 0 

        # Check each other fish to see if it is close enough to avoid 
        for boid in boids: 
            difference = self.position - boid.position 
            distance = np.linalg.norm(difference) 

            # Sum the weighted directions necessary to move directly away from # each nearby fish on the next turn. 
            if distance > 0 and distance < self.avoidanceDistance:
                difference /= distance # Normalize the direction 
                dDelta += difference / distance # Weight by inverse distance 
                totalNearby += 1 

        # Calculate the velocity necessary to steer away from the avoidance point 
        # on the next update (finish the weighted average) 
        if totalNearby > 0: 
            dDelta /= totalNearby 

            # Normalize the velocity away from avoidance point to get the 
            # direction necessary to accelerate in to avoid collision. 
            magDelta = np.linalg.norm(dDelta) 
            if magDelta > 0: 
                dDelta /= magDelta 
            else: 
                dDelta = np.zeros(2) 

        return dDelta

    def behave(self, boids): 
        alignment = self.align(boids) * self.aWeight 
        cohesion = self.cohese(boids) * self.cWeight 
        avoidance = self.avoid(boids) * self.sWeight 
        self.delta = self.delta + alignment + cohesion + avoidance
