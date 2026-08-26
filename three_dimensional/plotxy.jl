# using Plots
using CairoMakie
# using Measures
# using PGFPlotsX   # backend
using JLD2




@load "./data/angles_X_expectation_max_length_lmax5.jld2" angles_list ZZ_exp_list_list max_psum_length_list_list l_list

marker_list = [:circle,:rect,:diamond,:utriangle,:dtriangle,:xcross, :star5]
# 创建图像（对应 figsize）
fig = Figure(size=(800, 400),fontsize = 15)
ax = Axis(fig[1,1],xlabel = "θ/π",ylabel = "X expectation")

# 主曲线
for (l_idx,l) in enumerate(l_list)
    scatterlines!(angles_list ./ π, ZZ_exp_list_list[l_idx],
            marker=marker_list[l_idx],
          label="$(l)×$(l)×$(l)",
          linewidth=2)
end
# 竖线位置
x_pos_list = [0.25,0.75,1.25,1.75,2.25,2.75]





for x in x_pos_list
  scatterlines!([x, x], [-1.5, 1.5],
          color=:gray,
          linestyle=:dash,
          label=nothing)
end
analytic_2d = (cos.(2*angles_list)+abs.((cos.(2*angles_list))))./(2*cos.(angles_list))
scatterlines!(angles_list ./ π, analytic_2d,
        marker=marker_list[1],
      label="Analytic (2D)",
      linewidth=2)

axislegend(
  ax
)
# 保存与显示
file_name = "./figures/X_string_exp_$(l_list[end])l_3D.png"
save(file_name,fig)
run(`open $(file_name)`)