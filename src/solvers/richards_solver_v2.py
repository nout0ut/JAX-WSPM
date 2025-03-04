# src/solvers/richards_solver.py
import jax
import jax.numpy as jnp
from jax import jit, vmap, lax
import time
from pathlib import Path
from typing import Dict, Any, Tuple
from tqdm import tqdm
from functools import partial
from jax.experimental.sparse import BCOO
from jax import debug

import os, psutil
from datetime import datetime
import gc


from ..models.exponential import exponential_model
from ..models.boundary import (
    extract_boundary_nodes, 
    get_boundary_condition, 
    apply_dirichlet_bcs,
    shape_functions
)
from ..numerics.gauss import gauss_triangle
from ..numerics.assembly_re_v2 import assemble_global_matrices_sparse_re
from ..mesh.loader import load_and_validate_mesh
from .jax_linear_solver import JAXSparseSolver
from .solver_types import solve_system


def track_memory(message):
    """Track memory usage at specific points."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    timestamp = datetime.now().strftime('%H:%M:%S')
    print(f"[{timestamp}] {message}")
    print(f"    RSS Memory: {memory_info.rss / 1024 / 1024:.2f} MB")
    print(f"    VMS Memory: {memory_info.vms / 1024 / 1024:.2f} MB")    


class RichardsSolver:
    """Solver for the Richards equation using mixed form finite elements."""
    
    def __init__(self, config: 'SimulationConfig'):
        """Initialize the solver with configuration."""
        self.config = config
        self.setup_solver()
    
    def setup_solver(self):
        """Set up the solver components."""
        # Load and validate mesh
        self.points, self.triangles, self.mesh_info = load_and_validate_mesh(
            self.config.mesh.mesh_size,
            self.config.mesh.mesh_dir,
            self.config.test_case
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
        # Initial pressure head and domain length
        self.pressure_head = -15.24 * jnp.ones(self.nnt)
        self.L = 15.24  # Domain length
        
        # Calculate initial boundary conditions based on test case
        if self.bc_info['type'] == 'richards_only':
            # For Test1, Test2, Test3
            x_top = self.points[self.boundary_nodes.top][:, 0]
            eps0_top = jnp.exp(self.config.exponential.alpha0 * 
                             self.pressure_head[self.boundary_nodes.top])
            self.hupper = self.bc_info['upper_bc'](x_top, self.L, 
                                                 self.config.exponential.alpha0,
                                                 eps0_top)
    
    @partial(jit, static_argnums=(0,))
    def adapt_time_step(self, dt: float, iter_count: int) -> float:
        """Adapt time step based on iteration count."""
        time_params = self.config.time
        return lax.cond(
            iter_count <= time_params.m_it,
            lambda _: time_params.lambda_amp * dt,
            lambda _: lax.cond(
                iter_count <= time_params.M_it,
                lambda _: dt,
                lambda _: time_params.lambda_red * dt,
                None
            ),
            None
        )
    
    @partial(jit, static_argnums=(0,))
    def solve_timestep(self, pressure_head_n: jnp.ndarray, thetan_0: jnp.ndarray, dt: float) -> Tuple[jnp.ndarray, float, int]:
        """Solve one time step of Richards equation."""
        def body_fun(carry):
            pressure_head_m, pressure_head_n, err, iter_count = carry
            
            # Calculate soil properties
            Capacity_m, Konduc_m, thetan_m = vmap(exponential_model, in_axes=(0, None))(
                pressure_head_m, self.config.exponential.to_array())
            
            # Assemble matrices
            Global_matrix, Global_source = assemble_global_matrices_sparse_re(
                self.triangles, self.nnt, self.points,
                thetan_m, thetan_0, pressure_head_m,
                Konduc_m, Capacity_m, self.quad_points, self.weights, dt
            )
                     
            matrix_dense = Global_matrix.todense()
            
            if self.bc_info['type'] == 'richards_only':
                # Apply top boundary condition
                matrix_dense, Global_source = apply_dirichlet_bcs(
                    matrix_dense,
                    Global_source,
                    self.boundary_nodes.top, 
                    self.hupper
                )
                
                
                # Apply bottom/other boundary conditions (constant pressure head)
                matrix_dense, Global_source = apply_dirichlet_bcs(
                    matrix_dense,
                    Global_source,
                    self.boundary_nodes.bottom,
                    jnp.full_like(self.boundary_nodes.bottom, -15.24)
                )
            
            # Calculate matrix sum with explicit nse
            total_entries = self.triangles.shape[0] * 9
            Global_matrix = BCOO.fromdense(matrix_dense, nse=total_entries)
            
            pressure_head, convergence = solve_system(
                matrix=Global_matrix,
                rhs=Global_source,
                x0=pressure_head_m,
                solver_config=self.config.solver
            )

            err = jnp.linalg.norm(pressure_head - pressure_head_m) / jnp.linalg.norm(pressure_head)
            # Clear caches 
            jax.clear_caches()
            return pressure_head, pressure_head_n, err, iter_count + 1

        def cond_fun(carry):
            _, _, err, iter_count = carry
            return jnp.logical_and(err >= 1e-6, iter_count < 100)

        initial_carry = (pressure_head_n, pressure_head_n, jnp.inf, 0)
        final_carry = jax.lax.while_loop(cond_fun, body_fun, initial_carry)
        pressure_head, _, err, iter_count = final_carry
        jax.clear_caches()
        return pressure_head, err, iter_count
    
    # @partial(jit, static_argnums=(0,))
    # @jit
#     def solve_timestep(self, pressure_head_n: jnp.ndarray, thetan_0: jnp.ndarray, dt: float) -> Tuple[jnp.ndarray, float, int]:
#             """Solve one time step of Richards equation using traditional while loop."""

#             # track_memory("Start of Richards solve")

#             # Initialize variables
#             pressure_head_m = pressure_head_n.copy()
#             iter_count = 0
#             max_iterations = 100
#             tolerance = 1e-5
#             stop = False


#             while not stop:
#                 # track_memory("Start of iteration")

#                 # Get soil properties using original van_genuchten_model

#                 Capacity_m, Konduc_m, thetan_m = vmap(exponential_model, in_axes=(0, None))(
#                 pressure_head_m, self.config.exponential.to_array())

#                 # Assemble matrices using original function
#                 # track_memory("Before Richards assembly")
#                 Global_matrix, Global_source = assemble_global_matrices_sparse_re(
#                     self.triangles,
#                     self.nnt,
#                     self.points,
#                     thetan_m,
#                     thetan_0,
#                     pressure_head_m,
#                     Konduc_m,
#                     Capacity_m,
#                     self.quad_points,
#                     self.weights,
#                     dt
#                 )
#                 # track_memory("After Richards assembly")


#                 # Apply boundary conditions if needed
#                 matrix_dense = Global_matrix.todense()
#                 # track_memory("Afetr matrix_dense")

#                 # track_memory("Before BCs")
#                 if self.bc_info['type'] == 'richards_only':
#                     # Apply top boundary condition
#                     #print("Apply top boundary condition")
#                     matrix_dense, Global_source = apply_dirichlet_bcs(
#                         matrix_dense,
#                         Global_source,
#                         self.boundary_nodes.top, 
#                         self.hupper
#                     )


#                     # Apply bottom/other boundary conditions (constant pressure head)
#                     matrix_dense, Global_source = apply_dirichlet_bcs(
#                         matrix_dense,
#                         Global_source,
#                         self.boundary_nodes.bottom,
#                         jnp.full_like(self.boundary_nodes.bottom, -15.24)
#                     )
#                 # track_memory("After BCs")

#                 # Calculate matrix sum with explicit nse
#                 total_entries = self.triangles.shape[0] * 9
#                 Global_matrix = BCOO.fromdense(matrix_dense, nse=total_entries)
#                 pressure_head = jnp.linalg.solve(matrix_dense, Global_source)
#                 # track_memory("After BCOO")

#                 # Solve the linear system using original solve_system function
#                 # track_memory("Before linear system solve")
#                 pressure_head_new, convergence = solve_system(
#                     matrix=Global_matrix,
#                     rhs=Global_source,
#                     x0=pressure_head_m,
#                     solver_config=self.config.solver
#                 )
#                 # track_memory("After linear system solve")

#                 # Calculate error and update
#                 # track_memory("Before err calcu")
#                 error = jnp.linalg.norm(pressure_head_new - pressure_head_m)
#                 # track_memory("Afetr err calcu")
#                 pressure_head_m = pressure_head_new
#                 # track_memory("Afetr update")
#                 iter_count += 1

#                 # Check stopping criteria
#                 if error < tolerance or iter_count >= max_iterations:
#                     stop = True

#                 # Clear caches as in original code
#                 jax.clear_caches()
#                 # track_memory("End of iteration")

#             # track_memory("End of Richards solve")
#             return pressure_head_new, error, iter_count


    def solve(self) -> Dict[str, Any]:
        """Solve the Richards equation for the full simulation time."""
        simulation_start = time.perf_counter()
        
        # Initialize storage for results
        all_pressure = []
        all_theta = []
        all_times = []
        all_iterations = []
        all_errors = []
        all_dt = []
        
        # Initialize simulation variables
        pressure_head_n = self.pressure_head
        current_time = 0.0
        dt = float(self.config.time.dt_init)  # Convert to float
        
        # Progress bar
        pbar = tqdm(total=float(self.config.time.Tmax),
                   desc='Simulation Progress',
                   unit='time units')
        
        # Time stepping loop
        while current_time < self.config.time.Tmax:
            
            _, _, thetan_0 = vmap(exponential_model, in_axes=(0, None))(
                pressure_head_n, self.config.exponential.to_array())
            # Solve current time step
            pressure_head, error, iter_count = self.solve_timestep(
                pressure_head_n, thetan_0, dt)
            
            # Adapt time step and convert to float
            dt_new = float(self.adapt_time_step(dt, iter_count))
            # without adaptivity   
            #dt_new = float(dt)
            
            
            # Calculate water content
            _, _, theta = vmap(exponential_model, in_axes=(0, None))(
                pressure_head, self.config.exponential.to_array())
            
            # Store results
            all_pressure.append(pressure_head)
            all_theta.append(theta)
            all_times.append(current_time)
            all_iterations.append(int(iter_count))  # Convert to int
            all_errors.append(float(error))         # Convert to float
            all_dt.append(dt_new)                   # Already float
            
            # Update for next time step
            pressure_head_n = pressure_head
            current_time += dt_new
            dt = dt_new
            
            # Update progress bar with float values
            pbar.update(dt_new)
            pbar.set_description(
                f'Time: {float(current_time):.2f}, Error: {float(error):.2e}, '
                f'Iterations: {int(iter_count)}, dt: {dt_new:.2e}'
            )
        
        pbar.close()
        simulation_time = time.perf_counter() - simulation_start
        print(f"simulation time: {simulation_time}")
        
        # Prepare results dictionary
        results = {
            'pressure_head': jnp.array(all_pressure),
            'theta': jnp.array(all_theta),
            'times': jnp.array(all_times),
            'iterations': jnp.array(all_iterations),
            'errors': jnp.array(all_errors),
            'dt_values': jnp.array(all_dt),
            'points': self.points,
            'triangles': self.triangles,
            'simulation_time': simulation_time,
            'mesh_info': self.mesh_info,
            'config': self.config
        }
        # Calculate and add mass balance results
        return results
