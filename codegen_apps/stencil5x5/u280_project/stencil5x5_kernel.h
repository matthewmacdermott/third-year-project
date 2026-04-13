#ifndef STENCIL5X5_KERNEL_H
#define STENCIL5X5_KERNEL_H

void kernel_populate(ACC<float> &u) {
  u(0,0) = myfun();
}

void kernel_initialguess(ACC<float> &u) {
  u(0,0) = 0.0;
}

void stencil5x5_kernel_stencil(const ACC<float> &u,
                                ACC<float> &u2) {
  // 5x5 averaging stencil: compute average of 25 points
  // Weight for each point: 1/25 = 0.04
  const float weight = 0.04f;  // 1/25
  
  float sum = 0.0f;
  
  // Row -2
  sum += u(-2,-2) * weight;
  sum += u(-2,-1) * weight;
  sum += u(-2, 0) * weight;
  sum += u(-2, 1) * weight;
  sum += u(-2, 2) * weight;
  
  // Row -1
  sum += u(-1,-2) * weight;
  sum += u(-1,-1) * weight;
  sum += u(-1, 0) * weight;
  sum += u(-1, 1) * weight;
  sum += u(-1, 2) * weight;
  
  // Row 0
  sum += u( 0,-2) * weight;
  sum += u( 0,-1) * weight;
  sum += u( 0, 0) * weight;
  sum += u( 0, 1) * weight;
  sum += u( 0, 2) * weight;
  
  // Row 1
  sum += u( 1,-2) * weight;
  sum += u( 1,-1) * weight;
  sum += u( 1, 0) * weight;
  sum += u( 1, 1) * weight;
  sum += u( 1, 2) * weight;
  
  // Row 2
  sum += u( 2,-2) * weight;
  sum += u( 2,-1) * weight;
  sum += u( 2, 0) * weight;
  sum += u( 2, 1) * weight;
  sum += u( 2, 2) * weight;
  
  u2(0,0) = sum;
}

void kernel_copy(const ACC<float> &u2, ACC<float> &u) {
  u(0,0) = u2(0,0);
}

#endif //STENCIL5X5_KERNEL_H
