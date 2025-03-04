# src/solvers/coupled_solver.py
import jax
import jax.numpy as jnp
from jax import jit, vmap, lax
import time, psutil

import os
from datetime import datetime
import gc

from pathlib import Path
from typing import Dict, Any, Tuple
from tqdm import tqdm
from functools import partial
from jax.experimental.sparse import BCOO
from .solver_types import solve_system
from jax.experimental import sparse

from ..models.van_genuchten import van_genuchten_model

from ..models.boundary import get_boundary_condition, extract_boundary_nodes, apply_neumann_bcs, apply_cauchy_bcs_sparse

from ..models.transport_utilities import calculate_fluxes_and_dispersion_FE
from ..numerics.gauss import gauss_triangle
from ..numerics.assembly_re_v2 import assemble_global_matrices_sparse_re
from ..numerics.assembly_solute import assemble_global_matrices_solute
from ..mesh.loader import load_and_validate_mesh
# from .jax_linear_solver import JAXSparseSolver

    
def track_memory(message):
    """Track memory usage at specific points."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    timestamp = datetime.now().strftime('%H:%M:%S')
    print(f"[{timestamp}] {message}")
    print(f"    RSS Memory: {memory_info.rss / 1024 / 1024:.2f} MB")
    print(f"    VMS Memory: {memory_info.vms / 1024 / 1024:.2f} MB")    



class CoupledSolver:
    """Solver for coupled water flow and solute transport equations."""
    
    def __init__(self, config: 'SimulationConfig'):
        """Initialize the solver with configuration."""
        self.config = config
        self.setup_solver()
    
    def setup_solver(self):
        self.points, self.triangles, self.mesh_info = load_and_validate_mesh(
            self.config.mesh.mesh_size,
            self.config.mesh.mesh_dir,
            self.config.test_case  # Add this parameter
        )
        
        # Get boundary nodes
        self.boundary_nodes = extract_boundary_nodes(self.points, self.config.test_case)  
        # Initialize numerical integration
        self.quad_points, self.weights = gauss_triangle(3)
        self.ksi_1d = jnp.array([-1/jnp.sqrt(3), 1/jnp.sqrt(3)])
        self.w_1d = jnp.array([1.0, 1.0])
        
        # Get boundary condition function
        self.bc_info = get_boundary_condition(self.config.test_case)
                
        # Initialize solver parameters
        self.nnt = len(self.points)
        self.initialize_problem()
    
    def initialize_problem(self):
        """Initialize the problem variables."""
        # Initial pressure head
        self.pressure_head = -1.3 * jnp.ones(self.nnt)
        
        # Initial solute concentration
        self.solute = jnp.ones(self.nnt) * self.config.solute.c_init
        

    @partial(jit, static_argnums=(0,))
    def solve_richards_step(self, pressure_head_n: jnp.ndarray, thetan_0: jnp.ndarray, dt: float) -> Tuple[jnp.ndarray, float, int]:
        """Solve one time step of Richards equation using JAX while_loop."""

        def body_fun(carry):
            pressure_head_m, err, iter_count = carry

            # Get soil properties
            Capacity_m, Konduc_m, thetan_m = vmap(van_genuchten_model, in_axes=(0, None))(
                pressure_head_m, self.config.van_genuchten.to_array())

            # Assemble matrices
            Global_matrix, Global_source = assemble_global_matrices_sparse_re(
                self.triangles,
                self.nnt,
                self.points,
                thetan_m,
                thetan_0,
                pressure_head_m,
                Konduc_m,
                Capacity_m,
                self.quad_points,
                self.weights,
                dt
            )

            # Apply boundary conditions if needed
            if self.bc_info['type'] == 'coupled':
                neumann_value = self.bc_info['water_flux']['neumann_flux']
                Global_source = apply_neumann_bcs(
                    Global_source,
                    neumann_value,
                    self.boundary_nodes.neumann,
                    self.points,
                    self.ksi_1d,
                    self.w_1d
                )

            # Solve linear system
            pressure_head_new, convergence = solve_system(
                matrix=Global_matrix,
                rhs=Global_source,
                x0=pressure_head_m,
                solver_config=self.config.solver
            )

            # Calculate error
            err = jnp.linalg.norm(pressure_head_new - pressure_head_m)

            return pressure_head_new, err, iter_count + 1

        def cond_fun(carry):
            _, err, iter_count = carry
            return jnp.logical_and(err >= 1e-4, iter_count < 100)

        initial_carry = (pressure_head_n, jnp.inf, 0)
        final_carry = jax.lax.while_loop(cond_fun, body_fun, initial_carry)
        pressure_head_new, err, iter_count = final_carry

        jax.clear_caches()
        return pressure_head_new, err, iter_count
    
    
    
    
    
    
    @partial(jit, static_argnums=(0,))
    def solve_transport_step(self, 
                           solute_n: jnp.ndarray,
                           pressure_head: jnp.ndarray,
                           theta: jnp.ndarray,
                           theta_n: jnp.ndarray,
                           water_flux_x: jnp.ndarray,
                           water_flux_z: jnp.ndarray,
                           abs_q: float,
                           D_xx: jnp.ndarray,
                           D_xz: jnp.ndarray,
                           D_zz: jnp.ndarray,
                           dt: float) -> jnp.ndarray:
        """Solve one time step of solute transport equation."""

        
        # Assemble matrices
        Global_stiff, Global_mass = assemble_global_matrices_solute(
            self.triangles, self.nnt, self.points,
            theta, theta_n, water_flux_x, water_flux_z,
            D_xx, D_xz, D_zz,
            self.quad_points, self.weights
        )
        jax.clear_caches()
        Global_source = (1/dt) * (Global_mass @ solute_n)
        
        # Apply BCs using sparse format
        Global_stiff, Global_source = apply_cauchy_bcs_sparse(
            Global_stiff, Global_source,
            self.bc_info['solute']['cauchy_flux'],
            self.bc_info['solute']['inlet_conc'],
            self.boundary_nodes.cauchy,
            self.ksi_1d, self.w_1d, self.points
        )

        jax.clear_caches()
        
        # Form system matrix while keeping sparse format
        Global_matrix = ((1/dt) * Global_mass + Global_stiff)
        
        
        # In solve_transport_step:
        solute, convergence = solve_system(
            matrix=Global_matrix,
            rhs=Global_source,
            x0=solute_n,
            solver_config=self.config.solver
        )
        jax.clear_caches()
        
        return solute
    
    @partial(jit, static_argnums=(0,))
    def adapt_time_step(self,
                       dt: float,
                       iter_count: int) -> Tuple[float, float, float]:
        """Adapt time step based on iterations and stability criteria."""

        
        # Adjust time step based on iteration count
        dt_richards = lax.cond(
            iter_count <= self.config.time.m_it,
            lambda _: self.config.time.lambda_amp * dt,
            lambda _: lax.cond(
                iter_count <= self.config.time.M_it,
                lambda _: dt,
                lambda _: self.config.time.lambda_red * dt,
                None
            ),
            None
        )
        
        dt_new = jnp.minimum(dt_richards, 0.5)
        dt_new = jnp.maximum(dt_richards, 1e-6)
        return dt_new
    
    def solve(self) -> Dict[str, Any]:
        """Solve the coupled system for the full simulation time."""
        simulation_start = time.perf_counter()
        
        # Initialize storage
        all_pressure = []
        all_theta = []
        all_solute = []
        all_times = []
        all_iterations = []
        all_errors = []
        all_dt = []
        all_Pe = []
        all_Cr = []
        
        # Initialize simulation variables
        pressure_head_n = self.pressure_head
        # _, _, theta_n = vmap(van_genuchten_model, in_axes=(0, None))(
        #         pressure_head_n, self.config.van_genuchten.to_array())
        solute_n = self.solute
        current_time = 0.0
        dt = self.config.time.dt_init
        
        # Progress bar
        pbar = tqdm(total=float(self.config.time.Tmax),
                   desc='Simulation Progress',
                   unit='time units')
        
        while current_time < self.config.time.Tmax:
            # Solve Richards equation
            # Get water content at previous time
            _, _, theta_n = vmap(van_genuchten_model, in_axes=(0, None))(
                pressure_head_n, self.config.van_genuchten.to_array())
            
            pressure_head, error, iter_count = self.solve_richards_step(
                pressure_head_n, theta_n, dt)
            
            # Calculate water content and fluxes
            fluxes_disp = calculate_fluxes_and_dispersion_FE(
                self.points, self.triangles, pressure_head,
                self.config.van_genuchten.to_array(),
                self.config.solute.DL,
                self.config.solute.DT,
                self.config.solute.Dm
            )
             
            water_flux_x, water_flux_z, abs_q, D_xx, D_xz, D_zz, theta = fluxes_disp
            
            
            
            # Solve transport equation
            solute = self.solve_transport_step(
                solute_n, pressure_head, theta, theta_n, water_flux_x, water_flux_z, abs_q, D_xx, D_xz, D_zz, dt) 
            
            # Adapt. time step
            dt_new= self.adapt_time_step(
                dt, iter_count)
            # Without adapt.
            # dt_new = dt
            
            # Store results
            all_pressure.append(pressure_head)
            all_theta.append(theta)
            all_solute.append(solute)
            all_times.append(current_time)
            all_iterations.append(iter_count)
            all_errors.append(error)
            all_dt.append(dt_new)

            
            # Update for next time step
            pressure_head_n = pressure_head
            solute_n = solute
            current_time += float(dt)
            dt = dt_new
            
            # Update progress bar
            pbar.update(float(dt))  # Convert dt to float before updating
            pbar.set_description(
                f'Time: {float(current_time):.2f}, Error: {float(error):.2e}, '
                f'Iterations: {int(iter_count)}, dt: {float(dt):.2e}, '  # Also convert dt here
                
            )
        
        pbar.close()
        simulation_time = time.perf_counter() - simulation_start
        
        # Prepare results
        results = {
            'pressure_head': jnp.array(all_pressure),
            'theta': jnp.array(all_theta),
            'solute': jnp.array(all_solute),
            'times': jnp.array(all_times),
            'iterations': jnp.array(all_iterations),
            'errors': jnp.array(all_errors),
            'dt_values': jnp.array(all_dt),
            'Pe_values': jnp.array(all_Pe),
            'Cr_values': jnp.array(all_Cr),
            'points': self.points,
            'triangles': self.triangles,
            'final_pressure': pressure_head,
            'final_theta': theta,
            'final_solute': solute,
            'simulation_time': simulation_time,
            'mesh_info': self.mesh_info,
            'config': self.config
        }
        
        return results
