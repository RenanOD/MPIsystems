using Test

include("Kreader.jl")

iter = 0

problems = [
  "aug2d";
  "aug2dc";
  "aug2dcqp";
  "aug2dqp";
  "aug3d";
  "aug3dc";
  "aug3dcqp";
  "aug3dqp";
  "dual1";
  "dual2";
  "dual3";
  "dual4";
  "dualc1";
  #"dualc2";
  "dualc5";
  "dualc8";
  "aug2d";
  "aug2dc";
  "aug2dcqp";
  "aug2dqp";
  "aug3d";
  "aug3dc";
  "aug3dcqp";
  "aug3dqp";
  "cvxqp1_s";
  "cvxqp1_m";
  "cvxqp1_l";
  "cvxqp2_s";
  "cvxqp2_m";
  "cvxqp2_l";
  "cvxqp3_s";
  "cvxqp3_m";
  "cvxqp3_l";
  "genhs28";
  "gouldqp2";
  "gouldqp3";
  "hs118";
  "hs21";
  "hs21mod";
  "hs268";
  "hs35";
  "hs35mod";
  "hs51";
  "hs52";
  "hs53";
  "hs76";
  "hues-mod";
  "huestis";
  "ksip";
  "liswet1";
  "liswet10";
  "liswet11";
  "liswet12";
  "liswet2";
  "liswet3";
  "liswet4";
  "liswet5";
  "liswet6";
  "liswet7";
  "liswet8";
  "liswet9";
  "lotschd";
  "mosarqp1";
  "mosarqp2";
  "powell20";
  "primal1";
  "primal2";
  "primal3";
  "primal4";
  "primalc1";
  "primalc2";
  "primalc5";
  "primalc8";
  "qpcblend";
  "qpcboei1";
  "qpcboei2";
  "qpcstair";
  "s268";
  "stcqp1";
  "stcqp2";
  "tame";
  "ubh1";
  "yao";
  "zecevic2";
]

notbig = []
for (i, p) in enumerate(problems)
  cd("data/$p/3x3/iter_$iter")
  (rho, delta, H, J, Z, X, rhs) = read_blocks(iter)
  if size(J,2) < 8000
    push!(notbig, i)
  end
  for i in 1:4
    cd("..")
  end
end

@testset "Testa os sistemas K1 e K2 contra o K35" begin
  i = 0
  for p in problems[notbig]
    i+=1
    println("Problem $i")
    iter = 0
    cd("data/$p/3x3/iter_$iter")
    K1, rhs1 = assembleK1(iter)
    K2, rhs2 = assembleK2(iter)
    K35, rhs35 = assembleK35(iter)
    for i in 1:4
      cd("..")
    end

    sol1 = K1\rhs1
    sol2 = K2\rhs2
    sol35 = K35\rhs35

    dy1 = sol1
    dy2 = sol2[end-size(K1)[1]+1:end]
    dy35 = sol35[size(K2)[1]-size(K1)[1]+1:size(K2)[1]]

    dx2 = sol2[1:end-size(K1)[1]]
    dx35 = sol35[1:size(K2)[1]-size(K1)[1]]

    @test norm(dy1 - dy35) < 1e-4
    @test norm(dy2 - dy35) < 1e-4
    @test norm(dx2 - dx35) < 1e-4
  end
end
