using Plots
using Random
struct params
    mu_k::Float64
    sigma_k::Float64
    w_k::Float64
    mu_g::Float64
    sigma_g::Float64
    c_rep::Float64
    dt::Float64
    point_n::Int64
end

mutable struct fields
    R_val::Vector{Float64}
    U_val::Vector{Float64}
    R_grad::Vector{Float64}
    U_grad::Vector{Float64}
end
struct params
    mu_k::Float64
    sigma_k::Float64
    w_k::Float64
    mu_g::Float64
    sigma_g::Float64
    c_rep::Float64
    dt::Float64
    point_n::Int64
end
params() = params(4.0, 1.0, 0.022, 0.6, 0.15, 1.0, 0.1, 500)
fields(point_n) = fields(zeros(point_n), zeros(point_n), zeros(point_n*2), zeros(point_n*2))

function create_params_and_field()
    p = params()
    field = fields(p.point_n)
    return p, field
end

function init(point_n, points)
    for i in 0:point_n-1
        points[begin+i*2] = (rand()-0.5)*12
        points[begin+i*2+1] = (rand()-0.5)*12
    end
    return points
end

function add_xy!(a, i,x, y, c)
    a[begin+i*2] += x*c
    a[begin+i*2+1] += y*c
end

function repulsion_f(x, c_rep)
    t = max(1.0-x, 0.0)
    return [0.5*c_rep*t*t, -c_rep*t]
end
function fast_exp(x)
    t = 1.0 + x/32.0
    for i in 1:5
        t *= t
    end
    return t
end
function peak_f(x, mu, sigma, w=1.0)
    t = (x-mu)/sigma
    y = w/fast_exp(t*t)
    return [y, -2.0 * t * y / sigma]
end

function compute_fields(params, fields,points)
    fill!(fields.R_val, repulsion_f(0.0, params.c_rep)[1])
    fill!(fields.U_val, peak_f(0.0, params.mu_k, params.sigma_k, params.w_k)[1])
    fill!(fields.R_grad, 0.0)
    fill!(fields.U_grad, 0.0)
    for i in 0:params.point_n-2
        for j in (i+1):params.point_n-1
            rx = points[begin+i*2] - points[begin+j*2]
            ry = points[begin+i*2+1] - points[begin+j*2+1]
            r = sqrt(rx*rx + ry*ry) + 1e-20
            rx /= r
            ry /= r
            if r <1.0
                (R, dR) = repulsion_f(r, params.c_rep)
                add_xy!(fields.R_grad, i, rx, ry, dR)
                add_xy!(fields.R_grad, j, rx, ry, -dR)
                fields.R_val[begin+i] += R
                fields.R_val[begin+j] += R
            end
            (K, dK) = peak_f(r, params.mu_k, params.sigma_k, params.w_k)
            add_xy!(fields.U_grad, i, rx, ry, dK)
            add_xy!(fields.U_grad, j, rx, ry, -dK)
            fields.U_val[begin+i] += K
            fields.U_val[begin+j] += K
        end
    end

end


function step(params, fields, points)
    compute_fields(params, fields, points)
    total_E = 0.0
    for i in 0:params.point_n-1
        (G, dG) = peak_f(fields.U_val[begin+i], params.mu_g, params.sigma_g)
        vx = dG*fields.U_grad[begin+i*2]   - fields.R_grad[begin+i*2]
        vy = dG*fields.U_grad[begin+i*2+1] - fields.R_grad[begin+i*2+1]
        add_xy!(points, i, vx, vy, params.dt)
        total_E += fields.R_val[begin+i] - G
    end
    return (total_E/params.point_n)
end
world_width = 1
width = 500
height = 500
scale = (width/world_width, height/world_width)
p, field =  create_params_and_field()
points = init(p.point_n, zeros(p.point_n*2+2))
# an (x,y) point is in points as points[begin+i*2], points[begin+i*2+1]

sct = scatter(aspect_ratio=:equal)
x_arr = zeros(p.point_n)
y_arr = zeros(p.point_n)
r_arr = zeros(p.point_n)
@gif for i=1:2000
    for j=1:5
        step(p, field, points)
    end
    for i in 0:p.point_n-1
        x_arr[begin+i] = points[begin+i*2]
        y_arr[begin+i] = points[begin+i*2+1]
        r_arr[begin+i] = p.c_rep / (field.R_val[begin+i] *5.0)
    end
   scatter(x_arr, y_arr, markersize=r_arr.*20, markerstrokewidth=1,markercolor="white" , legend=false, color="black")
end every 10
