/// @file

#include "program.hpp"

#define X_MIN           -1.0f
#define X_MAX           1.0f
#define LENGTH          (float)(X_MAX-X_MIN)
#define NUM_POINTS      41
#define DX              (float)(LENGTH/NUM_POINTS)
#define KERNEL_FILE1     "../../kernel/thekernel1.cl"
#define KERNEL_FILE2     "../../kernel/thekernel2.cl"

// OpenCL queue
queue*  q1                = new queue();

// Kernels
kernel* k1                = new kernel();
kernel* k2                = new kernel();

point4* position          = new point4(NUM_POINTS);                             // Positions of nodes.
color4* color             = new color4(NUM_POINTS);                             // Particle colors.
float4* velocity          = new float4(NUM_POINTS);                             // Velocities of nodes.
float4* acceleration      = new float4(NUM_POINTS);                             // Accelerations of nodes.

point4* position_int      = new point4(NUM_POINTS);                             // Position (intermediate).
float4* velocity_int      = new float4(NUM_POINTS);                             // Velocity (intermediate).
float4* acceleration_int  = new float4(NUM_POINTS);                             // Acceleration (intermediate).

float4* gravity           = new float4(NUM_POINTS);                             // Gravity.
float4* stiffness         = new float4(NUM_POINTS);                             // Stiffness.
float4* resting           = new float4(NUM_POINTS);                             // Resting.
float4* friction          = new float4(NUM_POINTS);                             // Friction.
float4* mass              = new float4(NUM_POINTS);                             // Mass.

int1* index_PC            = new int1(NUM_POINTS);                               // Center particle.
int1* index_PR            = new int1(NUM_POINTS);                               // Right particle.
int1* index_PL            = new int1(NUM_POINTS);                               // Left particle.

float4* freedom           = new float4(NUM_POINTS);                              // Freedom/constrain flag.

float1* dt                = new float1(1);                                      // Time step.
float   simulation_time;                                                        // Simulation time.
int     time_step_number;                                                       // Time step index.

void setup()
{
  int i;
  float x;
  char buffer[100];

  // Cross-section area, volume density, Young's modulus, viscosity
  float A = 1e-4;
  float rho = 1e3;
  float E = 1e5;
  float mu = 500.0;

  // Model parameters (mass, gravity, stiffness, damping)
  float m = rho*A*DX;
  float g = 10.0f;
  float k = E*A/DX;
  float c = mu*A*DX;

  printf("Simulation parameters: k=%f, m=%f, c=%f\n", k, m, c);
  // Print header of output file
  snprintf(buffer, sizeof buffer, "# Simulation parameters: k=%f, m=%f, c=%f\n",
          k, m, c);
  write_file("out.csv", buffer);

  // Time step
  dt->x[0] = 0.5*sqrt(m/k);

  // Initializing OpenCL queue...
  q1->init();

  // Setting kernel #1 parameters and initializing...
  k1->source_file = KERNEL_FILE1;
  k1->size = NUM_POINTS;
  k1->dimension = 1;
  k1->init();

  // Setting kernel #2 parameters and initializing...
  k2->source_file = KERNEL_FILE2;
  k2->size = NUM_POINTS;
  k2->dimension = 1;
  k2->init();

  //////////////////////////////////////////////////////////////////////////////
  //////////////////////////Preparing data arrays... ///////////////////////////
  //////////////////////////////////////////////////////////////////////////////
  x = X_MIN;

  for (i = 0; i < NUM_POINTS; i++)
  {
    // Setting initial position components...
    position->x[i] = x;
    position->y[i] = 0.0f;
    position->z[i] = 0.0f;
    position->w[i] = 1.0f;

    gravity->x[i] = 0.0f;
    gravity->y[i] = -g;
    gravity->z[i] = 0.0f;
    gravity->w[i] = 1.0f;

    stiffness->x[i] = k;
    stiffness->y[i] = k;
    stiffness->z[i] = k;
    stiffness->w[i] = 1.0f;

    // Setting resting distance...
    resting->x[i] = DX;
    resting->y[i] = DX;
    resting->z[i] = DX;
    resting->w[i] = 1.0f;

    friction->x[i] = c;
    friction->y[i] = c;
    friction->z[i] = c;
    friction->w[i] = 1.0f;

    mass->x[i] = m;
    mass->y[i] = m;
    mass->z[i] = m;
    mass->w[i] = 1.0f;

    // Setting initial color components (RGBA)
    color->r[i] = 1.0f;
    color->g[i] = 0.0f;
    color->b[i] = 0.0f;
    color->a[i] = 1.0f;

    index_PC->x[i] =  i;

    freedom->x[i] = 1.0f;
    freedom->y[i] = 1.0f;
    freedom->z[i] = 1.0f;
    freedom->w[i] = 1.0f;

    if ((i != 0) && (i != (NUM_POINTS - 1)))   // When on bulk:
    {
      index_PR->x[i] = (i + 1);
      index_PL->x[i] = (i - 1);
    }
    else   // When on extremes:
    {
      gravity->x[i] = 0.0f;
      gravity->y[i] = 0.0f;
      gravity->z[i] = 0.0f;
      gravity->w[i] = 1.0f;

      freedom->x[i] = 0.0f;
      freedom->y[i] = 0.0f;
      freedom->z[i] = 0.0f;
      freedom->w[i] = 0.0f;
    }

    if (i == 0)   // When on left extreme:
    {
      index_PR->x[i] = (i + 1);
      index_PL->x[i] = index_PC->x[i];
    }

    if (i == (NUM_POINTS - 1))   // When on right extreme:
    {
      index_PR->x[i] = index_PC->x[i];
      index_PL->x[i] = (i - 1);
    }

    x += DX;
  }

  // Print info on time step (critical DT for stability)
  float cDT = sqrt(m/k);
  printf("Critical DT = %f\n", cDT);
  printf("Simulation DT = %f\n", dt->x[0]);

  // Set initial time to zero
  simulation_time = 0.0f;
  time_step_number = 0;

  // Initializing kernel variables...
  position->init();
  position_int->init();
  color->init();
  velocity->init();
  velocity_int->init();
  acceleration->init();
  acceleration_int->init();
  gravity->init();
  stiffness->init();
  resting->init();
  friction->init();
  mass->init();
  index_PC->init();
  index_PR->init();
  index_PL->init();
  freedom->init();
  dt->init();

  // Setting kernel arguments for kernel #1...
  position->set(k1, 0);
  color->set(k1, 1);
  position_int->set(k1, 2);
  velocity->set(k1, 3);
  velocity_int->set(k1, 4);
  acceleration->set(k1, 5);
  acceleration_int->set(k1, 6);
  gravity->set(k1, 7);
  stiffness->set(k1, 8);
  resting->set(k1, 9);
  friction->set(k1, 10);
  mass->set(k1, 11);
  index_PR->set(k1, 12);
  index_PL->set(k1, 13);
  freedom->set(k1, 14);
  dt->set(k1,15);

  // Setting kernel arguments for kernel #2...
  position->set(k2, 0);
  color->set(k2, 1);
  position_int->set(k2, 2);
  velocity->set(k2, 3);
  velocity_int->set(k2, 4);
  acceleration->set(k2, 5);
  acceleration_int->set(k2, 6);
  gravity->set(k2, 7);
  stiffness->set(k2, 8);
  resting->set(k2, 9);
  friction->set(k2, 10);
  mass->set(k2, 11);
  index_PR->set(k2, 12);
  index_PL->set(k2, 13);
  freedom->set(k2, 14);
  dt->set(k2,15);
}

void post_process(queue* q, point4* position)
{
  // Save vertical position of midpoint every 10 time steps
  if(time_step_number%50 == 0)
  {
    float t;
    float x[4*NUM_POINTS];
    float y;
    cl_int err;

    t = simulation_time;

    // Read vector of positions at time t
    err = clEnqueueReadBuffer(q->thequeue, position->buffer,
                              CL_TRUE, 0, 4*sizeof(cl_float)*NUM_POINTS,
                              x, 0, NULL, NULL);
    if(err < 0)
    {
      printf("\nError:  %s\n", get_error(err));
      exit(EXIT_FAILURE);
    }

    // Vertical position of midpoint
    y = x[4*(NUM_POINTS-1)/2+1];

    // Format output string and write to CSV file
    char buffer [100];
    snprintf(buffer, sizeof buffer, "%f,%f\n", t, y);
    printf("T = %f\n", simulation_time);
    write_file("out.csv", buffer);
  }
}

void loop()
{
  // Pushing kernel arguments to device memory (kernel #1)...
  position->push(q1, k1, 0);
  color->push(q1, k1, 1);
  position_int->push(q1, k1, 2);
  velocity->push(q1, k1, 3);
  velocity_int->push(q1, k1, 4);
  acceleration->push(q1, k1, 5);
  acceleration_int->push(q1, k1, 6);
  gravity->push(q1, k1, 7);
  stiffness->push(q1, k1, 8);
  resting->push(q1, k1, 9);
  friction->push(q1, k1, 10);
  mass->push(q1, k1, 11);
  index_PR->push(q1, k1, 12);
  index_PL->push(q1, k1, 13);
  freedom->push(q1, k1, 14);
  dt->push(q1, k1, 15);

  // Executing kernel #1 and waiting for its termination...
  k1->execute(q1, WAIT);

  // Executing kernel #2 and waiting for its termination...
  k2->execute(q1, WAIT);

  // Post-process simulation data
  post_process(q1, position);

  // Popping kernel arguments for kernel #2...
  position->pop(q1, k2, 0);
  color->pop(q1, k2, 1);
  position_int->pop(q1, k2, 2);
  velocity->pop(q1, k2, 3);
  velocity_int->pop(q1, k2, 4);
  acceleration->pop(q1, k2, 5);
  acceleration_int->pop(q1, k2, 6);
  gravity->pop(q1, k2, 7);
  stiffness->pop(q1, k2, 8);
  resting->pop(q1, k2, 9);
  friction->pop(q1, k2, 10);
  mass->pop(q1, k2, 11);
  index_PR->pop(q1, k2, 12);
  index_PL->pop(q1, k2, 13);
  freedom->pop(q1, k2, 14);
  dt->pop(q1, k2, 15);

  // Plotting current configuration of the rope...
  plot(position, color, STYLE_POINT);

  // Update simulation time
  simulation_time += dt->x[0];
  time_step_number += 1;
}

void terminate()
{
  delete position;
  delete position_int;
  delete color;
  delete velocity;
  delete velocity_int;
  delete acceleration;
  delete acceleration_int;
  delete gravity;
  delete stiffness;
  delete resting;
  delete friction;
  delete mass;
  delete index_PC;
  delete index_PL;
  delete index_PR;
  delete freedom;
  delete dt;

  delete q1;
  delete k1;
  delete k2;

  printf("All done!\n");
}
