using LinearAlgebra
using SparseArrays
using Printf
using CSV
using MatrixMarket
using DelimitedFiles

function read_blocks(iter)
  rd = CSV.read(@sprintf("rho_delta_%d.dat", iter), datarow = 1)[1]
  rho = rd[1]
  delta = rd[2]

  H = MatrixMarket.mmread(@sprintf("H+rhoI_%d.mtx", iter))
  H = -H
  J = MatrixMarket.mmread(@sprintf("J1J2_%d.mtx", iter))
  Z = MatrixMarket.mmread(@sprintf("Zsqrt_%d.mtx", iter))
  # trabalhar com z em vez de Z
  X = MatrixMarket.mmread(@sprintf("S_%d.mtx", iter))
  # trabalhar com x em vez de X
  rhs = readdlm(@sprintf("rhs_%d.rhs", iter))

  return rho, delta, H, J, Z, X, rhs
end

# Notation for the problem
# min x^T*H*x + c^Tx
# s.t. Jx = b
# X has the original variables in its main diagonal
# Z has the slack variables in its main diagonal
# rhs is the right hand side of the system
# rho and delta are constants designed to improve the system's stability

# Assemble
# K1 = -[J*(H + rho*I + X^{-1} Z)^{-1}*J' + delta*I]
# and return corresponding right hand side rhs.

function assembleK1(iter, quadratic = false)
  (rho, delta, H, J, Z, X, rhs) = read_blocks(iter)
  (m, n) = size(J)
  if quadratic
    K = -(J*( sparse(inv(Matrix(H + tril(H,-1)' + Z'*(X \ Z))))) *J' + delta*sparse(Matrix(1.0I, m, m)))
  else
    K = -(J*( sparse(inv(Matrix(rho*sparse(Matrix(1.0I, n, n)) + Z'*(X \ Z))))) *J' + delta*sparse(Matrix(1.0I, m, m)))
  end
  ns = size(Z, 1)       # number of slack variables
  nn = size(H, 1) - ns  # number of original variables
  rhs[1:nn+ns] = rhs[1:nn+ns] - Z' * (X \ rhs[nn+ns+m+1:nn+ns+m+ns])
  rhs = rhs[1:nn+ns+m]
  if quadratic
    temp = J*sparse(inv(Matrix(H + tril(H,-1)' + Z'*(X \ Z))))
  else
    temp = J*sparse(inv(Matrix(rho*sparse(Matrix(1.0I, n, n)) + Z'*(X \ Z))))
  end
  sl = size(temp)[2]
  rhs = rhs[sl+1:end] - temp*rhs[1:sl]
  return K, rhs
end

# tentar usar isso para deixar de usar a matriz com rho e ver se volta a funcionar!
# teste = Z'*(X \ Z)
# for i in 1:length(teste.nzval)
#    teste.nzval[i] = 1/teste.nzval[i]
#end

# Assemble
# K2 = [ H + rho*I + X^{-1} Z     J'     ]
#      [         J              -delta*I ]
# and return corresponding right hand side rhs.

function assembleK2(iter, quadratic = false)
  (rho, delta, H, J, Z, X, rhs) = read_blocks(iter)

  (m, n) = size(J)
  ns = size(Z, 1)      # number of slack variables
  nn = size(H, 1) - ns  # number of original variables
  if quadratic
    K = [ H + tril(H,-1)' + Z' * (X \ Z)       J';
          J  -delta * sparse(Matrix(1.0I, m, m)) ]
  else
    K = [ rho*sparse(Matrix(1.0I, n, n)) + Z' * (X \ Z)    J';
          J              -delta * sparse(Matrix(1.0I, m, m)) ]
  end
  # reducing the rhs:
  rhs[1:nn+ns] = rhs[1:nn+ns] - Z' * (X \ rhs[nn+ns+m+1:nn+ns+m+ns])
  rhs = rhs[1:nn+ns+m]

  return K, rhs
end


# Assemble
#      [ H + rho*I     J'    -I ]
# K3 = [     J     -delta*I     ]
#      [     Z                X ]
# and return corresponding right hand side rhs.

function assembleK3(iter)
  II = Z'
  II[findall(II .> 0)] .= 1.0

  (m, n) = size(J)
  ns = size(Z, 1);      # number of slack variables.
  n = size(H, 1) - ns;  # number of original variables.

  K = [ H + tril(H,-1)'      J'                -II            ;
        J  -delta * sparse(Matrix(1.0I, m, m))  spzeros(m, ns);
        Z.^2                 spzeros(ns, m)      X            ]

  rhs[end-ns+1:end] = Z[:, n+1:end] * rhs[end-ns+1:end]

  return K, rhs
end
