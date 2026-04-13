#ifndef STAR5PT_KERNEL_H
#define STAR5PT_KERNEL_H

void kernel_populate(ACC<float> &u) {
  u(0,0) = myfun();
}

void kernel_initialguess(ACC<float> &u) {
  u(0,0) = 0.0;
}

void star5pt_kernel_stencil(const ACC<float> &u,
                             ACC<float> &u2) {
  // 5-point star stencil: center + 4 cardinal directions
  // Weight for each point: 1/5 = 0.2
  const float weight = 0.2f;  // 1/5
  
  float sum = 0.0f;
  
  // Cardinal directions (North, South, East, West) + Center
  sum += u( 0,-2) * weight;  // North
  sum += u(-2, 0) * weight;  // West
  sum += u( 0, 0) * weight;  // Center
  sum += u( 2, 0) * weight;  // East
  sum += u( 0, 2) * weight;  // South
  
  u2(0,0) = sum;
}

void kernel_copy(const ACC<float> &u2, ACC<float> &u) {
  u(0,0) = u2(0,0);
}

#endif //STAR5PT_KERNEL_H
