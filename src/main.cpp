#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <ctime>
#include <cmath>

// ---- Structs to match GLSL (std430 rules: align on 16-byte boundaries) ----

struct Material {
    int type;      // 1 = Lambertian, 2 = Metal, 3 = Dielectric
    float fuzz;    // Metal only
    float ir;      // Dielectric only (index of refraction)
    float pad0;    // Padding
    float albedo[4]; // pad to vec4
};

struct Sphere {
    float center[4]; // x, y, z, pad
    float radius;
    float pad1, pad2, pad3; // Pad to vec4
    Material mat;
};

// ---- Random number generation ----
std::mt19937 rng((unsigned)time(nullptr));
float randf(float a, float b) {
    std::uniform_real_distribution<float> dist(a, b);
    return dist(rng);
}

// ---- Random scene generation ----
void generate_random_scene(std::vector<Sphere>& spheres) {
    // Ground sphere
    {
        Sphere s = {};
        s.center[0] = 0.0f; s.center[1] = -1000.0f; s.center[2] = 0.0f;
        s.radius = 1000.0f;
        s.mat.type = 1; // Lambertian
        s.mat.albedo[0] = 0.5f;
        s.mat.albedo[1] = 0.5f;
        s.mat.albedo[2] = 0.5f;
        s.mat.albedo[3] = 0.0f;
        spheres.push_back(s);
    }
    // Random small spheres
    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            float choose_mat = randf(0.0f, 1.0f);
            glm::vec3 center(a + 0.9f * randf(0.0f, 1.0f), 0.2f, b + 0.9f * randf(0.0f, 1.0f));
            if (glm::length(center - glm::vec3(4, 0.2f, 0)) > 0.9f) {
                Sphere s = {};
                s.center[0] = center.x;
                s.center[1] = center.y;
                s.center[2] = center.z;
                s.radius = 0.2f;
                if (choose_mat < 0.8f) { // diffuse
                    s.mat.type = 1;
                    s.mat.albedo[0] = randf(0.0f, 1.0f) * randf(0.0f, 1.0f);
                    s.mat.albedo[1] = randf(0.0f, 1.0f) * randf(0.0f, 1.0f);
                    s.mat.albedo[2] = randf(0.0f, 1.0f) * randf(0.0f, 1.0f);
                    s.mat.albedo[3] = 0.0f;
                }
                else if (choose_mat < 0.95f) { // metal
                    s.mat.type = 2;
                    s.mat.albedo[0] = randf(0.5f, 1.0f);
                    s.mat.albedo[1] = randf(0.5f, 1.0f);
                    s.mat.albedo[2] = randf(0.5f, 1.0f);
                    s.mat.albedo[3] = 0.0f;
                    s.mat.fuzz = randf(0.0f, 0.5f);
                }
                else { // glass
                    s.mat.type = 3;
                    s.mat.albedo[0] = 1.0f;
                    s.mat.albedo[1] = 1.0f;
                    s.mat.albedo[2] = 1.0f;
                    s.mat.albedo[3] = 0.0f;
                    s.mat.ir = 1.5f;
                }
                spheres.push_back(s);
            }
        }
    }
    // Three large spheres
    {
        Sphere s1 = {};
        s1.center[0] = 0.0f; s1.center[1] = 1.0f; s1.center[2] = 0.0f;
        s1.radius = 1.0f; s1.mat.type = 3; s1.mat.ir = 1.5f;
        s1.mat.albedo[0] = s1.mat.albedo[1] = s1.mat.albedo[2] = 1.0f;
        spheres.push_back(s1);

        Sphere s2 = {};
        s2.center[0] = -4.0f; s2.center[1] = 1.0f; s2.center[2] = 0.0f;
        s2.radius = 1.0f; s2.mat.type = 1;
        s2.mat.albedo[0] = 0.4f; s2.mat.albedo[1] = 0.2f; s2.mat.albedo[2] = 0.1f;
        spheres.push_back(s2);

        Sphere s3 = {};
        s3.center[0] = 4.0f; s3.center[1] = 1.0f; s3.center[2] = 0.0f;
        s3.radius = 1.0f; s3.mat.type = 2;
        s3.mat.albedo[0] = 0.7f; s3.mat.albedo[1] = 0.6f; s3.mat.albedo[2] = 0.5f;
        s3.mat.fuzz = 0.0f;
        spheres.push_back(s3);
    }
}

// ---- Shader check helpers ----
void checkShaderCompile(GLuint shader) {
    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetShaderInfoLog(shader, 512, nullptr, infoLog);
        std::cerr << "Shader compilation failed:\n" << infoLog << std::endl;
    }
}

void checkProgramLink(GLuint program) {
    GLint success;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetProgramInfoLog(program, 512, nullptr, infoLog);
        std::cerr << "Program linking failed:\n" << infoLog << std::endl;
    }
}

// ---- Main entry point ----
int main() {
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    const int image_width = 1200;
    const int image_height = 675;
    GLFWwindow* window = glfwCreateWindow(image_width, image_height, "Ray Tracer", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW" << std::endl;
        glfwTerminate();
        return -1;
    }

    // ---- Random scene creation ----
    std::vector<Sphere> spheres;
    generate_random_scene(spheres);
    int num_spheres = (int)spheres.size();

    // ---- Compute shader ----
    const char* computeShaderSource = R"(
        #version 430 core

        layout(local_size_x = 8, local_size_y = 8) in;
        layout(rgba32f, binding = 0) uniform image2D outputImage;

        uniform vec2 resolution;
        uniform int num_spheres;
        uniform float time;
        uniform int samples_per_pixel;

        struct Material {
            int type;
            float fuzz;
            float ir;
            float pad0;
            vec4 albedo;
        };
        struct Sphere {
            vec4 center; // xyz, pad
            float radius;
            float pad1, pad2, pad3;
            Material mat;
        };

        layout(std430, binding = 0) buffer SphereBuffer {
            Sphere spheres[];
        };

        struct ray { vec3 origin; vec3 direction; };
        vec3 at(ray r, float t) { return r.origin + t*r.direction; }

        struct hit_record {
            vec3 p;
            vec3 normal;
            float t;
            bool front_face;
            int sphere_index;
        };

        const int LAMBERTIAN = 1;
        const int METAL = 2;
        const int DIELECTRIC = 3;

        float random(vec2 st) {
            return fract(sin(dot(st, vec2(12.9898, 78.233))) * 43758.5453123);
        }
        vec3 random_unit_vector(vec2 seed) {
            float a = random(seed) * 6.2831853;
            float z = random(seed+vec2(1.0)) * 2.0 - 1.0;
            float r = sqrt(1.0 - z*z);
            return vec3(r*cos(a), r*sin(a), z);
        }
        vec3 reflect(vec3 v, vec3 n) { return v - 2.0*dot(v,n)*n; }
        bool refract(vec3 v, vec3 n, float ni_over_nt, out vec3 refracted) {
            vec3 uv = normalize(v);
            float dt = dot(uv, n);
            float discriminant = 1.0 - ni_over_nt*ni_over_nt*(1.0-dt*dt);
            if (discriminant > 0.0) {
                refracted = ni_over_nt*(uv-n*dt)-n*sqrt(discriminant);
                return true;
            }
            return false;
        }
        float schlick(float cosine, float ref_idx) {
            float r0 = (1.0-ref_idx)/(1.0+ref_idx);
            r0 = r0*r0;
            return r0 + (1.0-r0)*pow((1.0-cosine),5.0);
        }

        bool hit_sphere(Sphere s, ray r, float t_min, float t_max, out hit_record rec) {
            vec3 oc = r.origin - s.center.xyz;
            float a = dot(r.direction, r.direction);
            float half_b = dot(oc, r.direction);
            float c = dot(oc, oc) - s.radius*s.radius;
            float discriminant = half_b*half_b - a*c;

            if (discriminant < 0.0) return false;
            float sqrtd = sqrt(discriminant);
            float root = (-half_b - sqrtd)/a;
            if (root < t_min || root > t_max) {
                root = (-half_b + sqrtd)/a;
                if (root < t_min || root > t_max) return false;
            }
            rec.t = root;
            rec.p = at(r, rec.t);
            vec3 outward_normal = (rec.p - s.center.xyz) / s.radius;
            rec.front_face = dot(r.direction, outward_normal) < 0.0;
            rec.normal = rec.front_face ? outward_normal : -outward_normal;
            return true;
        }

        bool hit_world(ray r, float t_min, float t_max, out hit_record rec) {
            hit_record temp_rec;
            bool hit_anything = false;
            float closest_so_far = t_max;
            int hit_sphere_index = -1;
            for (int i = 0; i < num_spheres; ++i) {
                if (hit_sphere(spheres[i], r, t_min, closest_so_far, temp_rec)) {
                    hit_anything = true;
                    closest_so_far = temp_rec.t;
                    hit_sphere_index = i;
                    rec = temp_rec;
                }
            }
            rec.sphere_index = hit_sphere_index;
            return hit_anything;
        }

        bool scatter(ray r_in, hit_record rec, vec2 seed, out vec3 attenuation, out ray scattered) {
            Material mat = spheres[rec.sphere_index].mat;
            if (mat.type == LAMBERTIAN) {
                vec3 scatter_direction = rec.normal + random_unit_vector(seed);
                if (length(scatter_direction)<1e-8) scatter_direction = rec.normal;
                scattered.origin = rec.p;
                scattered.direction = scatter_direction;
                attenuation = mat.albedo.xyz;
                return true;
            } else if (mat.type == METAL) {
                vec3 reflected = reflect(normalize(r_in.direction), rec.normal);
                scattered.origin = rec.p;
                scattered.direction = reflected + mat.fuzz * random_unit_vector(seed);
                attenuation = mat.albedo.xyz;
                return dot(scattered.direction, rec.normal) > 0.0;
            } else if (mat.type == DIELECTRIC) {
                attenuation = vec3(1.0);
                float refraction_ratio = rec.front_face ? (1.0 / mat.ir) : mat.ir;
                vec3 unit_direction = normalize(r_in.direction);
                float cos_theta = min(dot(-unit_direction, rec.normal), 1.0);
                float sin_theta = sqrt(1.0 - cos_theta*cos_theta);
                bool cannot_refract = refraction_ratio * sin_theta > 1.0;
                vec3 direction;
                if (cannot_refract || schlick(cos_theta, mat.ir) > random(seed))
                    direction = reflect(unit_direction, rec.normal);
                else
                    refract(unit_direction, rec.normal, refraction_ratio, direction);
                scattered.origin = rec.p;
                scattered.direction = direction;
                return true;
            }
            return false;
        }

        vec3 ray_color(ray initial_ray, int max_depth, vec2 seed) {
            vec3 color = vec3(0.0);
            vec3 attenuation_product = vec3(1.0);
            ray current_ray = initial_ray;
            hit_record rec;
            for (int depth = 0; depth < max_depth; depth++) {
                if (hit_world(current_ray, 0.001, 1000.0, rec)) {
                    ray scattered; vec3 attenuation;
                    vec2 current_seed = seed + vec2(float(depth));
                    if (scatter(current_ray, rec, current_seed, attenuation, scattered)) {
                        attenuation_product *= attenuation;
                        current_ray = scattered;
                    } else {
                        color = vec3(0.0);
                        break;
                    }
                } else {
                    vec3 unit_direction = normalize(current_ray.direction);
                    float t = 0.5 * (unit_direction.y + 1.0);
                    color = attenuation_product * mix(vec3(1.0), vec3(0.5,0.7,1.0), t);
                    break;
                }
            }
            return color;
        }

        void main() {
            ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
            if (pixel.x >= resolution.x || pixel.y >= resolution.y) return;

            // Camera setup (adjusted for zoomed-in view)
            float aspect_ratio = resolution.x / resolution.y;
            float vfov = 20.0; // Vertical field of view in degrees
            float theta = vfov * 3.14159265359 / 180.0;
            float h = tan(theta / 2.0);
            float viewport_height = 2.0 * h;
            float viewport_width = aspect_ratio * viewport_height;
            float focal_length = 1.0;

            vec3 lookfrom = vec3(13.0, 2.0, 3.0);
            vec3 lookat = vec3(0.0, 0.0, 0.0);
            vec3 vup = vec3(0.0, 1.0, 0.0);
            vec3 w = normalize(lookfrom - lookat);
            vec3 u = normalize(cross(vup, w));
            vec3 v = cross(w, u);

            vec3 origin = lookfrom;
            vec3 horizontal = viewport_width * u;
            vec3 vertical = viewport_height * v;
            vec3 lower_left_corner = origin - 0.5 * horizontal - 0.5 * vertical - focal_length * w;

            const int max_depth = 20;
            vec3 color = vec3(0.0);
            vec2 pixel_f = vec2(pixel);

            for (int s = 0; s < samples_per_pixel; ++s) {
                vec2 seed = pixel_f + vec2(float(s) * 1234.567, float(s) * 7654.321);
                float u = (pixel.x + random(seed)) / (resolution.x - 1.0);
                float v = (pixel.y + random(seed + 17.0)) / (resolution.y - 1.0);

                ray r;
                r.origin = origin;
                r.direction = normalize(lower_left_corner + u * horizontal + v * vertical - origin);
                color += ray_color(r, max_depth, seed);
            }

            color /= float(samples_per_pixel);
            color = sqrt(color); // Gamma correction
            imageStore(outputImage, pixel, vec4(color, 1.0));
        }
    )";

    // ---- Compile compute shader ----
    GLuint computeShader = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(computeShader, 1, &computeShaderSource, nullptr);
    glCompileShader(computeShader);
    checkShaderCompile(computeShader);

    GLuint computeProgram = glCreateProgram();
    glAttachShader(computeProgram, computeShader);
    glLinkProgram(computeProgram);
    checkProgramLink(computeProgram);
    glDeleteShader(computeShader);

    // ---- Upload spheres to SSBO ----
    GLuint sphereSSBO;
    glGenBuffers(1, &sphereSSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, sphereSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, spheres.size() * sizeof(Sphere), spheres.data(), GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, sphereSSBO);

    // ---- Create output texture ----
    GLuint outputTexture;
    glGenTextures(1, &outputTexture);
    glBindTexture(GL_TEXTURE_2D, outputTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, image_width, image_height, 0, GL_RGBA, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glBindImageTexture(0, outputTexture, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);

    // ---- Run compute shader ----
    glUseProgram(computeProgram);
    glUniform1i(glGetUniformLocation(computeProgram, "samples_per_pixel"), 10);
    glUniform2f(glGetUniformLocation(computeProgram, "resolution"), (float)image_width, (float)image_height);
    glUniform1i(glGetUniformLocation(computeProgram, "num_spheres"), num_spheres);
    glUniform1f(glGetUniformLocation(computeProgram, "time"), (float)glfwGetTime());

    glDispatchCompute((image_width + 7) / 8, (image_height + 7) / 8, 1);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

    // ---- Read pixels from texture ----
    std::vector<float> pixels(image_width * image_height * 4);
    glBindTexture(GL_TEXTURE_2D, outputTexture);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, pixels.data());

    // ---- Save to PPM ----
    std::ofstream ppmFile("output.ppm");
    ppmFile << "P3\n" << image_width << " " << image_height << "\n255\n";
    for (int j = image_height - 1; j >= 0; --j) {
        for (int i = 0; i < image_width; ++i) {
            int idx = (j * image_width + i) * 4;
            int r = (int)(pixels[idx] * 255.0f);
            int g = (int)(pixels[idx + 1] * 255.0f);
            int b = (int)(pixels[idx + 2] * 255.0f);
            ppmFile << r << " " << g << " " << b << "\n";
        }
    }
    ppmFile.close();

    // ---- Clean up ----
    glDeleteTextures(1, &outputTexture);
    glDeleteProgram(computeProgram);
    glDeleteBuffers(1, &sphereSSBO);
    glfwDestroyWindow(window);
    glfwTerminate();

    std::cout << "Image rendered and saved as output.ppm\n";
    return 0;
}