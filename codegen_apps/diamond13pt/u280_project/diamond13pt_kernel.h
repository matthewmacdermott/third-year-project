#ifndef DIAMOND13PT_KERNEL_H
#define DIAMOND13PT_KERNEL_H

void kernel_populate(ACC<float> &u) {
  u(0,0) = myfun();
}

void kernel_initialguess(ACC<float> &u) {
  u(0,0) = 0.0;
}

void diamond13pt_kernel_stencil(const ACC<float> &u,
                                  ACC<float> &u2) {
  // 13-point diamond stencil: taxicab distance <= 2
  // Weight for each point: 1/13 ≈ 0.076923
  const float weight = 0.076923077f;  // 1/13
  
  float sum = 0.0f;
  
  // Distance 0 (center)
  sum += u( 0, 0) * weight;
  
  // Distance 1 (4 points)
  sum += u(-1, 0) * weight;
  sum += u( 1, 0) * weight;
  sum += u( 0,-1) * weight;
  sum += u( 0, 1) * weight;
  
  // Distance 2 (8 points)
  sum += u(-2, 0) * weight;
  sum += u( 2, 0) * weight;
  sum += u( 0,-2) * weight;
  sum += u( 0, 2) * weight;
  sum += u(-1,-1) * weight;
  sum += u(-1, 1) * weight;
  sum += u( 1,-1) * weight;
  sum += u( 1, 1) * weight;
  
  u2(0,0) = sum;
}

void kernel_copy(const ACC<float> &u2, ACC<float> &u) {
  u(0,0) = u2(0,0);
}

#endif //DIAMOND13PT_KERNEL_H
