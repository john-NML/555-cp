# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 John Leland

import numpy as np
import matplotlib.pyplot as plt
import random


def gen_neutron(): # Neutron generator to create initial neutron population based on nuSigmaf
    sigma_fission1 = 0.15
    sigma_fission2 = 0.12

    # calculate fission bin of the left region
    sigma_fission_total = sigma_fission1 + sigma_fission2
    left_region_bin = sigma_fission1/sigma_fission_total

    neutron = random.random()
    if neutron < left_region_bin: # if the random number is within the left region bin, spawn it uniformily in it
        return random.random()/2 * 100
    else: # otherwise spawn it uniformly in the right region
        return (random.random()/2+0.5) * 100
    
def left_region(): # Function holding cross section data for the left region
    sigma_absorption = 0.12
    sigma_scattering = 0.05
    sigma_fission = 0.15
    return sigma_absorption, sigma_scattering, sigma_fission

def right_region(): # Function holding cross section data for the right region
    sigma_absorption = 0.10
    sigma_scattering = 0.05
    sigma_fission = 0.12
    return sigma_absorption, sigma_scattering, sigma_fission


def cx_binning(sigmas): # Function to create collision type bins and total cross section
    sigma_absorption = sigmas[0]
    sigma_scattering = sigmas[1]
    sigma_fission = sigmas[2]
    sigma_total = sigma_scattering + sigma_absorption

    abs_bin = sigma_absorption/sigma_total
    sca_bin = sigma_scattering/sigma_total + abs_bin

    nu_fission = sigma_fission

    return sigma_total, abs_bin, sca_bin, nu_fission


class medium_data: # Class holding boundary information for the medium
    def __init__(self):
        self.left_bound = 0
        self.right_bound = 100
        self.interface = 50

medium = medium_data()

def region_finder(position): # Function to find the correct collision bins based on neutron position
    if position < medium.interface: # if the neutron is to the left of the interface return the cx data of the left region
        return cx_binning(left_region())
    else: # otherwise return right region cx data
        return cx_binning(right_region())
    
def mean_free_path(sigma_total): # calculates the distance the neutron travels
    return -1/sigma_total*np.log(random.random())

def neutron_path(position, direction): # Function to find neutron path to each boundary given the direction

    # if the neutron is traveling right and in the left region calculate the distance to the interface and right bound. Neutron cannot travel to the left bound
    if direction > 0: 
        if position < medium.interface:
            interface_dist = medium.interface - position
        else:
            interface_dist = np.inf
        
        right_bound_dist = medium.right_bound - position
        left_bound_dist = np.inf
    
    # if the neutron is traveling left and in the right region calculate the distance to the interface and left bound. Neutron cannot travel to the right bound
    elif direction < 0:
        if position > medium.interface:
            interface_dist = medium.interface - position
        else:
            interface_dist = np.inf

        left_bound_dist = medium.left_bound - position
        right_bound_dist = np.inf

    # if the direction is 0, the neutron is in limbo
    else:
        interface_dist = np.inf
        right_bound_dist = np.inf
        left_bound_dist = np.inf

    return interface_dist, right_bound_dist, left_bound_dist

def tally_bin(position,tally_bin_width): # Function for finding the tally bin given neutron position
    return int(position//tally_bin_width)

class tally: # Class for tracking collision, track length, and flux tallies
    def __init__(self, bins):
        # initialize the number bins, the bin width, and the tally arrays
        self.tally_bins = bins
        self.tally_bin_width = (medium.right_bound - medium.left_bound)/self.tally_bins
        self.track_length_tally = np.zeros(self.tally_bins)
        self.collision_tally = np.zeros(self.tally_bins)
        self.flux_tally = np.zeros(self.tally_bins)
        

    def track_length(self, position, traveled, direction, N): # Track length estimator
        # input the neutron start position, end position, direction, and current generation population
        # This function adds the neutron track length from left to right in the track length tally array
        self.population = N
        self.direction = direction
        self.previous_position = position
        self.current_position = traveled

        end_point = 1e-12

        # determine the left most and right most positions
        if self.previous_position == self.current_position:
            return
        elif self.current_position > self.previous_position:
            x0 = self.previous_position
            x1 = self.current_position
        else:
            x1 = self.previous_position
            x0 = self.current_position

        # find the start and end bins
        self.starting_tally = tally_bin(x0, self.tally_bin_width)
        self.ending_tally   = tally_bin(x1, self.tally_bin_width)

        # calculate the right edge of the current bin
        tally_edge = (self.starting_tally + 1) * self.tally_bin_width
        
        tally_position = x0
        current_tally = self.starting_tally
        
        # loop through adding the track length to every bin until the end position or end tally bin are reached
        while tally_position < x1 - end_point and 0 < current_tally < self.tally_bins:

            tally_right_edge = min(tally_edge, x1 - end_point) # right edge of current bin
            path_length_in_bin = tally_right_edge - tally_position # length traveled within the current bin

            if path_length_in_bin > 0: # as long as the length traveled isn't 0, add the track length
                # the track length is normalized by the current generation population and bin width
                self.track_length_tally[current_tally] += path_length_in_bin/abs(self.direction)/(self.population*self.tally_bin_width) 

            tally_position = tally_right_edge # increment the current tally position

            tally_edge += self.tally_bin_width # move the right edge to the next bin
            current_tally += 1 # increment the tally bin

    def collision(self, position, sigma_total): # Collision rate estimator
        # requires the ending position of the neutron and the total cross of the current region
        self.tally_bin = tally_bin(position, self.tally_bin_width) # find the tally bin the collision occurs in
        # add 1 collision normalized by the total cross section, current generation population, and bin width
        self.collision_tally[self.tally_bin] += 1/(sigma_total*self.population*self.tally_bin_width) 

    def flux(self, direction, start, end): # This is the "flux tally" estimator. I'm not sure how to interpret it but it was used in a previous class I took about neutron transport
        self.start_tally = tally_bin(start, self.tally_bin_width)
        self.end_tally = tally_bin(end, self.tally_bin_width)

        if self.end_tally > self.start_tally:
            self.flux_tally[self.starting_tally+1:self.ending_tally+1] += 1/abs(direction)/self.population
        else:
            self.flux_tally[self.end_tally+1:self.start_tally+1] += 1/abs(direction)/self.population


##### Constants or input variables
N = 10000
InactiveGenerations = 20
Generations = 20
k_generation = 1.2

##### Initialize tallies
bins = 200
tallies = tally(bins)    


NeutronBank = []
k_estimator = []
k_average = 0

for i in range(N): # Generate initial population of neutrons
    neutron = gen_neutron()
    NeutronBank.append(neutron)


##### Begin MC
for gen in range(InactiveGenerations + Generations):

    starting_neutron_pop = len(NeutronBank) # calculate number of neutrons
    FissionBank = []
    k=0
    Fissions = 0

    if gen == InactiveGenerations: # Reset tallies for active generations
        tallies = tally(bins)

    for n in range(len(NeutronBank)): # transport each neutron in the NeutronBank

        neutron_position = NeutronBank[n]

        neutron_lost = False
        u = 2*random.random()-1 # sample direction

        while not(neutron_lost):        

            sigma_total, abs_bin, sca_bin, nu_fission = region_finder(neutron_position) # find cx data 

            neutron_mfp = mean_free_path(sigma_total) # sample neutron distance

            neutron_distance = neutron_mfp*u # calculate horizontal distance

            interface_distance, right_bound_distance, left_bound_distance = neutron_path(neutron_position, u) # find distance to bounds

            neutron_path_to_surface = min([interface_distance, right_bound_distance, left_bound_distance], key=abs) # find closest bound

            neutron_path_length = min([neutron_distance, neutron_path_to_surface], key=abs) # determine if neutron hits interface or travels
            
            neutron_traveled = neutron_position + neutron_path_length

            tallies.track_length(neutron_position, neutron_traveled, u, starting_neutron_pop) # fill track length estimator
            tallies.flux(u, neutron_position, neutron_traveled)

            if neutron_traveled >= medium.right_bound or neutron_traveled <= medium.left_bound: # determine if neutron leaves the problem space
                neutron_lost = True
            elif abs(neutron_distance) > abs(neutron_path_to_surface): # if the neutron hits the interface place it on the other side and go back to the beginning
                if u > 0:
                    neutron_position = neutron_traveled + 1e-10
                else:
                    neutron_position = neutron_traveled - 1e-10
                continue
            else: # continue to collision physics
                new_neutron_position = neutron_traveled
            
            sigma_total, abs_bin, sca_bin, nu_fission = region_finder(new_neutron_position)
            tallies.collision(new_neutron_position, sigma_total) # fill collision rate estimator

            interaction = random.random()
            if interaction < abs_bin: # deterine if neutron is absorbed

                xn = nu_fission/(abs_bin*sigma_total) # lewis and miller page 352
                k += xn

                # find the number of neutrons produced per fission
                fission_neutrons = xn/k_generation
                fission_rand = random.random()

                Im = int(np.floor(fission_neutrons))
                Rm = fission_neutrons - Im

                if fission_rand > Rm:
                    neutrons_from_fission = Im
                else:
                    neutrons_from_fission = Im + 1
                
                for i in range(neutrons_from_fission): # add fission neutrons to FissionBank
                    FissionBank.append(new_neutron_position)
                
                neutron_lost = True

            else: # if the neutron scatters, sample new direction and then go back to the beginning
                neutron_position = new_neutron_position

                # Lewis and Miller page 341
                xi = random.random()
                w = 2*np.pi*xi
                cos_theta = 2*random.random()-1
                sin_theta = np.sqrt(1-cos_theta**2)
                cos_omega = np.cos(w)

                u = sin_theta*np.sqrt(1-u**2)*cos_omega + u*cos_theta

    k_generation = k/(starting_neutron_pop)

    print('GEN:', gen, ' Pop:', starting_neutron_pop, 'k=',k_generation)
    
    if gen > InactiveGenerations: # once in active generations, starting recording k
        k_estimator.append(k_generation)

    # Resampled_FissionBank = []
    # for i in range(len(FissionBank)-(len(FissionBank)-starting_neutron_pop)):
    #     idx = int(random.random()*len(FissionBank))
    #     Resampled_FissionBank.append(FissionBank[idx])
    NeutronBank = FissionBank # NeutronBank for next generation is the FissionBank of the previous



k_average = np.mean(k_generation) # calculate average multiplication constant
print('k average = ', k_average)

### Plot the flux estimators averaged by the number of active generations
x = np.linspace(0,100,tallies.tally_bins)

plt.plot(x,tallies.collision_tally/Generations)
plt.xlabel('Position (cm)')
plt.ylabel('Collision Flux Tally')
plt.show()
plt.close()

# plt.plot(x,tallies.flux_tally/Generations)
# plt.xlabel('Position (cm)')
# plt.ylabel('Flux Tally')
# plt.show()
# plt.close()

plt.plot(x,tallies.track_length_tally/Generations)
plt.xlabel('Position (cm)')
plt.ylabel('Track Length Flux Tally')
plt.show()
plt.close()

plt.plot(x,tallies.collision_tally/Generations, label='Collision')
plt.plot(x,tallies.track_length_tally/Generations, label='Track Length')
# plt.plot(x,tallies.flux_tally/Generations, label='Flux')
plt.xlabel('Position (cm)')
plt.ylabel('Flux')
plt.ylim([0,0.18])
plt.tight_layout()
plt.grid()
plt.legend()
plt.show()
plt.close()

