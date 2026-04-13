#ifndef DIAGONAL_X_KERNEL_H
#define DIAGONAL_X_KERNEL_H

void kernel_populate(ACC<float> &u) {
  u(0,0) = myfun();
}

void kernel_initialguess(ACC<float> &u) {
  u(0,0) = 0.0;
}

void diagonal_x_kernel_stencil(const ACC<float> &u,
                                 ACC<float> &u2) {
  // Diagonal X stencil: uses points along both diagonals
  // 9 unique points: 5 on main diagonal + 4 on anti-diagonal (center shared)
  // Weight for each point: 1/9 ≈ 0.111111
  const float weight = 0.111111111f;  // 1/9
  
  float sum = 0.0f;
  
  // Main diagonal (top-left to bottom-right): \
  sum += u(-2,-2) * weight;
  sum += u(-1,-1) * weight;
  sum += u( 0, 0) * weight;
  sum += u( 1, 1) * weight;
  sum += u( 2, 2) * weight;
  
  // Anti-diagonal (top-right to bottom-left): /
  sum += u(-2, 2) * weight;
  sum += u(-1, 1) * weight;
  // u(0,0) already counted
  sum += u( 1,-1) * weight;
  sum += u( 2,-2) * weight;
  
  u2(0,0) = sum;
}

void kernel_copy(const ACC<float> &u2, ACC<float> &u) {
  u(0,0) = u2(0,0);
}

#endif //DIAGONAL_X_KERNEL_H
