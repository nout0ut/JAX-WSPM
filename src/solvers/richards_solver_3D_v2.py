"""
3D Richards equation solver implementation.
"""
import jax
import jax.numpy as jnp
from jax import jit, vmap, lax
import time
from pathlib import Path
from typing import Dict, Any, Tuple
from tqdm import tqdm
from functools import partial
from jax.experimental.sparse import BCOO

import os, psutil
from datetime import datetime
import gc

from ..models.exponential import exponential_model
from ..models.boundary import (
    extract_boundary_nodes, 
    get_boundary_condition, 
    apply_dirichlet_bcs, apply_dirichlet_bcs_sparse, apply_all_bcs_sparse
)
from ..numerics.gauss_3D import gauss_tetrahedron
from ..numerics.assembly_re_3D import assemble_global_matrices_sparse_3D
from ..mesh.loader import load_and_validate_mesh
from .solver_types import solve_system


def track_memory(message):
    """Track memory usage at specific points."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    timestamp = datetime.now().strftime('%H:%M:%S')
    print(f"[{timestamp}] {message}")
    print(f"    RSS Memory: {memory_info.rss / 1024 / 1024:.2f} MB")
    print(f"    VMS Memory: {memory_info.vms / 1024 / 1024:.2f} MB")  



class RichardsSolver3D:
    """Solver for the 3D Richards equation using mixed form finite elements."""
    
    def __init__(self, config: 'SimulationConfig'):
        """Initialize the solver with configuration."""
        self.config = config
        self.setup_solver()
    
    def setup_solver(self):
        """Set up the solver components."""
        # Load and validate mesh (3D)
        self.points, self.tetrahedra, self.mesh_info = load_and_validate_mesh(
            self.config.mesh.mesh_size,
            self.config.mesh.mesh_dir,
            'Test3D',  # Specific for 3D mesh
            prefix=('p_3D', 't_3D')  # Use 3D mesh files _68921
        )
        
        # Get boundary nodes
        self.boundary_nodes = extract_boundary_nodes(self.points, 'Test3D')
        
        # Initialize numerical integration
        self.quad_points, self.weights = gauss_tetrahedron(4)  # Use tetrahedral quadrature
        
        # Get boundary condition function
        self.bc_info = get_boundary_condition('Test3D')

        # Initialize solver parameters
        self.nnt = len(self.points)
        self.initialize_problem()
    
    def initialize_problem(self):
        """Initialize the problem variables."""
        # Initial pressure head and domain length
        self.pressure_head = -15.24 * jnp.ones(self.nnt)
        self.L = 15.24  # Domain length
        
        # Calculate initial boundary conditions
        if self.bc_info['type'] == 'richards_only':
            x_top = self.points[self.boundary_nodes.top][:, 0]
            y_top = self.points[self.boundary_nodes.top][:, 1]
            eps0_top = jnp.exp(self.config.exponential.alpha0 * 
                             self.pressure_head[self.boundary_nodes.top])
            self.hupper = self.bc_info['upper_bc'](
                x_top, y_top, self.L, 
                self.config.exponential.alpha0,
                eps0_top
            )
    
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
    
    # # @partial(jit, static_argnums=(0,))
    # def solve_timestep(self, pressure_head_n: jnp.ndarray, thetan_0: jnp.ndarray, dt: float) -> Tuple[jnp.ndarray, float, int]:
    #     """Solve one time step of 3D Richards equation."""
    #     def body_fun(carry):
    #         pressure_head_m, pressure_head_n, err, iter_count = carry
            
    #         # Calculate soil properties
    #         Capacity_m, Konduc_m, thetan_m = vmap(exponential_model, in_axes=(0, None))(
    #             pressure_head_m, self.config.exponential.to_array())
            
    #         # Assemble matrices (3D version)
    #         track_memory("Before Richards assembly")
    #         Global_matrix, Global_source = assemble_global_matrices_sparse_3D(
    #             self.tetrahedra, self.nnt, self.points,
    #             thetan_m, thetan_0, pressure_head_m,
    #             Konduc_m, Capacity_m, self.quad_points, self.weights, dt
    #         )
    #         track_memory("After Richards assembly")
            
    #         matrix_dense = Global_matrix.todense()
            
    #         # Apply boundary conditions
    #         matrix_dense, Global_source = apply_dirichlet_bcs(
    #             matrix_dense,
    #             Global_source,
    #             self.boundary_nodes.top, 
    #             self.hupper
    #         )

    #         track_memory("After Apply BCs")
            
    #         # Apply bottom/other boundary conditions
    #         matrix_dense, Global_source = apply_dirichlet_bcs(
    #             matrix_dense,
    #             Global_source,
    #             self.boundary_nodes.bottom,
    #             jnp.full_like(self.boundary_nodes.bottom, -15.24)
    #         )
    #         track_memory("After Apply bottom BCs")
            
    #         # Convert back to sparse
    #         total_entries = self.tetrahedra.shape[0] * 16  # 4x4 matrices for tetrahedra
    #         Global_matrix = BCOO.fromdense(matrix_dense, nse=total_entries)
            
    #         # Solve system
    #         pressure_head, convergence = solve_system(
    #             matrix=Global_matrix,
    #             rhs=Global_source,
    #             x0=pressure_head_m,
    #             solver_config=self.config.solver
    #         )
            

    #         track_memory("After solve linear syst.")

    #         err = jnp.linalg.norm(pressure_head - pressure_head_m)
    #         jax.clear_caches()
    #         track_memory("After iterative step")
    #         return pressure_head, pressure_head_n, err, iter_count + 1

    #     def cond_fun(carry):
    #         _, _, err, iter_count = carry
    #         return jnp.logical_and(err >= 1e-4, iter_count < 30)
        
    #     initial_carry = (pressure_head_n, pressure_head_n, jnp.inf, 0)
    #     final_carry = jax.lax.while_loop(cond_fun, body_fun, initial_carry)
    #     pressure_head, _, err, iter_count = final_carry
    #     jax.clear_caches()
        
    #     return pressure_head, err, iter_count

    def solve_timestep(self, pressure_head_n: jnp.ndarray, thetan_0: jnp.ndarray, dt: float) -> Tuple[jnp.ndarray, float, int]:
        """Solve one time step of 3D Richards equation."""
        # Initialize variables
        pressure_head_m = pressure_head_n
        err = jnp.inf
        iter_count = 0
        
        # track_memory("Start of solve_timestep")
        
        while err >= 1e-4 and iter_count < 30:
            
            # Calculate soil properties
            Capacity_m, Konduc_m, thetan_m = vmap(exponential_model, in_axes=(0, None))(
                pressure_head_m, self.config.exponential.to_array())
            
            # Assemble matrices (3D version)
            Global_matrix, Global_source = assemble_global_matrices_sparse_3D(
                self.tetrahedra, self.nnt, self.points,
                thetan_m, thetan_0, pressure_head_m,
                Konduc_m, Capacity_m, self.quad_points, self.weights, dt
            )
            
            # Apply all boundary conditions directly on sparse matrix
            Global_matrix, Global_source = apply_all_bcs_sparse(
                Global_matrix,
                Global_source,
                self.boundary_nodes,
                self.hupper
            )
            
            # Solve system
            pressure_head, convergence = solve_system(
                matrix=Global_matrix,
                rhs=Global_source,
                x0=pressure_head_m,
                solver_config=self.config.solver
            )
            
            # Calculate error and update variables
            err = jnp.linalg.norm(pressure_head - pressure_head_m)
            pressure_head_m = pressure_head
            iter_count += 1
            
            jax.clear_caches()
        
        return pressure_head, err, iter_count
        
    def solve(self) -> Dict[str, Any]:
        """Solve the 3D Richards equation for the full simulation time."""
        
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
        dt = float(self.config.time.dt_init)
        
        # Progress bar
        pbar = tqdm(total=float(self.config.time.Tmax),
                   desc='3D Simulation Progress',
                   unit='time units')
        
        simulation_start = time.perf_counter()
        
        # Time stepping loop
        while current_time < self.config.time.Tmax:
            _, _, thetan_0 = vmap(exponential_model, in_axes=(0, None))(
                pressure_head_n, self.config.exponential.to_array())
            
            # Solve current time step
            pressure_head, error, iter_count = self.solve_timestep(
                pressure_head_n, thetan_0, dt)
            
            # Adapt time step
            dt_new = float(self.adapt_time_step(dt, iter_count))
            
            # Store results
            all_pressure.append(pressure_head)
            all_theta.append(thetan_0)
            all_times.append(current_time)
            all_iterations.append(int(iter_count))
            all_errors.append(float(error))
            all_dt.append(dt_new)
            
            # Update for next time step
            pressure_head_n = pressure_head
            current_time += dt_new
            dt = dt_new
            
            # Update progress bar
            pbar.update(dt_new)
            pbar.set_description(
                f'Time: {float(current_time):.2f}, Error: {float(error):.2e}, '
                f'Iterations: {int(iter_count)}, dt: {dt_new:.2e}'
            )
        
        pbar.close()
        simulation_time = time.perf_counter() - simulation_start
        
        # Prepare results dictionary
        results = {
            'pressure_head': jnp.array(all_pressure),
            'theta': jnp.array(all_theta),
            'times': jnp.array(all_times),
            'iterations': jnp.array(all_iterations),
            'errors': jnp.array(all_errors),
            'dt_values': jnp.array(all_dt),
            'points': self.points,
            'tetrahedra': self.tetrahedra,  # Note: using tetrahedra instead of triangles
            'final_theta': jnp.array(all_theta[-1]),
            'simulation_time': simulation_time,
            'mesh_info': self.mesh_info,
            'config': self.config
        }
        
        # Clear any remaining caches
        jax.clear_caches()
        gc.collect()
        
        return results
