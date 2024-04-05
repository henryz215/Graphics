#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <random>
#include "CImg_latest/CImg-3.2.7_pre081723/CImg.h"
#include "glm-master//glm/ext.hpp"
#include "glm-master/glm/glm.hpp"
#include "glm-master/glm/vec3.hpp"
#include "glm-master/glm/vec4.hpp"
#include "glm-master/glm/gtc/matrix_transform.hpp"
#include "tinyxml2-master/tinyxml2.h"
#include "tinyxml2-master/tinyxml2.cpp"

#define M_PI 3.14159265358979323846  
#define MAX_DEPTH 5

using namespace cimg_library;
using namespace tinyxml2;
using namespace glm;
using namespace std;

struct Ray {
    vec3 origin;
    vec3 direction;
};
struct SphericalAreaLight {
    vec3 center;    
    float radius;      
    vec3 color;        
    float intensity;  

    SphericalAreaLight(const vec3& c, float r, const vec3& col, float inten)
        : center(c), radius(r), color(col), intensity(inten) {}

    vec3 getRandomPointOnSurface() const {
        float u = static_cast<float>(rand()) / RAND_MAX;
        float v = static_cast<float>(rand()) / RAND_MAX;
        float theta = u * 2.0 * M_PI;
        float phi = acos(2.0 * v - 1.0);
        float x = sin(phi) * cos(theta);
        float y = sin(phi) * sin(theta);
        float z = cos(phi);
        vec3 randomDirection(x, y, z);

        return center + radius * randomDirection;
    }
};
struct Material {
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
    float shininess;
    float transparency;
    float refraction;
};
struct Sphere {
    vec3 center;
    float radius;
    vec3 color;
    Material material;
};
struct Triangle {
    vec3 v0, v1, v2;
    vec3 color;
    Material material;
};
struct Plane {
    vec3 normal;
    float distance;
    vec3 color;
    Material material;
};
struct DirectionalLight {
    vec3 direction;
    vec3 color;
};
class Scene {
public:
    Scene(int width, int height) : image(width, height, 1, 3) {}

    void addSphere(const Sphere& sphere) {
        spheres.push_back(sphere);
    }

    void addTriangle(const Triangle& triangle) {
        triangles.push_back(triangle);
    }

    void addPlane(const Plane& plane) {
        planes.push_back(plane);
    }

    void setDirectionalLight(const DirectionalLight& light) {
        dirLight = light;
    }
    void render() {
        const int width = image.width();
        const int height = image.height();
        const float aspectRatio = static_cast<float>(width) / height;
        const float fov = 45.0f;
        const int samples = 32;

        vec3 cameraPos(0, 0.25, 5);
        vec3 cameraTarget(0, 0, 0);
        vec3 cameraUp(0, 1, 0);
        mat4 view = lookAt(cameraPos, cameraTarget, cameraUp);
        mat4 projection = perspective(radians(fov), aspectRatio, 0.1f, 100.0f);
        mat4 viewProjection = projection * view;

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                vec3 colorSum(0.0f);

                for (int s = 0; s < samples; s++) {
                    Ray ray;
                    ray.origin = cameraPos;

                    float jitterX = static_cast<float>(rand()) / RAND_MAX - 0.5f;
                    float jitterY = static_cast<float>(rand()) / RAND_MAX - 0.5f;

                    ray.direction = normalize(vec3(x - width / 2 + jitterX, height / 2 - y + jitterY, -width / (2 * tan(radians(fov) / 2))));
                    ray.direction = normalize(vec3(view * vec4(ray.direction, 0.0)));

                    vec3 color = trace(ray);
                    colorSum += color;
                }

                // Average
                vec3 Color = colorSum / static_cast<float>(samples);
                image(x, y, 0, 0) = static_cast<unsigned char>(Color.r * 255);
                image(x, y, 0, 1) = static_cast<unsigned char>(Color.g * 255);
                image(x, y, 0, 2) = static_cast<unsigned char>(Color.b * 255);
            }
        }
    }

    const CImg<unsigned char>& getImage() const {
        return image;
    }

    void saveImage(const string& filename) {
        image.save(filename.c_str());
    }

    const vector<Sphere>& getSpheres() const {
        return spheres;
    }

    // Apply transformations to objects
    void applyTransformations(float translationX, float translationY, float translationZ, float rotationX, float rotationY, float rotationZ) {
        for (Sphere& sphere : spheres) {
            sphere.center += vec3(translationX, translationY, translationZ);

            glm::mat4 rotationMatrixX = glm::rotate(glm::mat4(1.0f), glm::radians(rotationX), glm::vec3(1, 0, 0));
            glm::mat4 rotationMatrixY = glm::rotate(glm::mat4(1.0f), glm::radians(rotationY), glm::vec3(0, 1, 0));
            glm::mat4 rotationMatrixZ = glm::rotate(glm::mat4(1.0f), glm::radians(rotationZ), glm::vec3(0, 0, 1));

            sphere.center = vec3(rotationMatrixX * rotationMatrixY * rotationMatrixZ * vec4(sphere.center, 1.0));
        }
        for (Triangle& triangle : triangles) {
            triangle.v0 += vec3(translationX, translationY, translationZ);
            triangle.v1 += vec3(translationX, translationY, translationZ);
            triangle.v2 += vec3(translationX, translationY, translationZ);

            glm::mat4 rotationMatrixX = glm::rotate(glm::mat4(1.0f), glm::radians(rotationX), glm::vec3(1, 0, 0));
            glm::mat4 rotationMatrixY = glm::rotate(glm::mat4(1.0f), glm::radians(rotationY), glm::vec3(0, 1, 0));
            glm::mat4 rotationMatrixZ = glm::rotate(glm::mat4(1.0f), glm::radians(rotationZ), glm::vec3(0, 0, 1));

            triangle.v0 = vec3(rotationMatrixX * rotationMatrixY * rotationMatrixZ * vec4(triangle.v0, 1.0));
            triangle.v1 = vec3(rotationMatrixX * rotationMatrixY * rotationMatrixZ * vec4(triangle.v1, 1.0));
            triangle.v2 = vec3(rotationMatrixX * rotationMatrixY * rotationMatrixZ * vec4(triangle.v2, 1.0));
        }
        for (Plane& plane : planes) {
            plane.normal += vec3(translationX, translationY, translationZ);

            glm::mat4 rotationMatrixX = glm::rotate(glm::mat4(1.0f), glm::radians(rotationX), glm::vec3(1, 0, 0));
            glm::mat4 rotationMatrixY = glm::rotate(glm::mat4(1.0f), glm::radians(rotationY), glm::vec3(0, 1, 0));
            glm::mat4 rotationMatrixZ = glm::rotate(glm::mat4(1.0f), glm::radians(rotationZ), glm::vec3(0, 0, 1));

            plane.normal = vec3(rotationMatrixX * rotationMatrixY * rotationMatrixZ * vec4(plane.normal, 1.0));
        }
    }

    vec3 generateRandomPointOnAreaLight() {
        // Assuming the area light source is a square of size 1 centered at the origin
        float randX = static_cast<float>(rand()) / RAND_MAX - 0.5f;
        float randY = static_cast<float>(rand()) / RAND_MAX - 0.5f;

        return vec3(randX, 1.0f, randY);
    }

private:
    CImg<unsigned char> image;
    vector<Sphere> spheres;
    vector<Triangle> triangles;
    vector<Plane> planes;
    DirectionalLight dirLight;

    glm::vec3 reflect(const glm::vec3& incident, const glm::vec3& normal) {
        return incident - 2.0f * glm::dot(incident, normal) * normal;
    }

    bool RefractedDirection(const vec3& incident, const vec3& normal, float refractionIndex, vec3& refracted) {
        float cosThetaI = glm::dot(incident, normal);
        float etaI = 1.0f;
        float etaT = refractionIndex;

        if (cosThetaI < 0.0f) {
            cosThetaI = -cosThetaI;
        }
        else {
            // swap the refractive indices
            swap(etaI, etaT);
        }

        float eta = etaI / etaT;
        float k = 1.0f - eta * eta * (1.0f - cosThetaI * cosThetaI);

        if (k < 0.0f) {
            // Internal reflection
            return false;
        }

        refracted = eta * incident + (eta * cosThetaI - sqrtf(k)) * normal;

        return true;
    }


    vec3 trace(const Ray& ray, int depth = 0) {
        if (depth > MAX_DEPTH) {
            return vec3(0.0f);
        }

        vec3 intersectionColor = vec3(0.0f);
        vec3 finalColor = vec3(0.0f);
        float closestT = numeric_limits<float>::max();
        int ShadowRays = 4; //  The number of rays that will be shot


        // Calculates with consideration of directional light (Spheres)
        for (const Sphere& sphere : spheres) {
            float t;
            if (intersectSphere(ray, sphere, t) && t < closestT) {
                Material mat = sphere.material;
                closestT = t;
                vec3 intersectionPoint = ray.origin + t * ray.direction;
                vec3 normal = normalize(intersectionPoint - sphere.center);
                vec3 viewDirection = normalize(ray.origin - intersectionPoint);


                if (mat.shininess > 0.0f) {
                    vec3 reflectedDir = reflect(ray.direction, normal);
                    Ray reflectedRay;
                    reflectedRay.origin = intersectionPoint + 0.001f * reflectedDir;
                    reflectedRay.direction = reflectedDir;

                    vec3 reflectionColor = trace(reflectedRay, depth + 1);

                    float shininess = 50.0f; // You can adjust the shininess value
                    vec3 reflectionDirection = reflect(ray.direction, normal);

                    // Combine reflection color with the intersection color
                    finalColor = intersectionColor + mat.specular * reflectionColor;
                }

                if (mat.transparency > 0.0f) {
                    vec3 transparencyRayOrigin = intersectionPoint - 0.001f * ray.direction;
                    vec3 transparencyRayDirection = ray.direction; // Assuming no refraction for simplicity

                    bool hasRefraction = RefractedDirection(ray.direction, normal, mat.refraction, transparencyRayDirection);

                    if (hasRefraction) {
                        Ray transparencyRay;
                        transparencyRay.origin = transparencyRayOrigin;
                        transparencyRay.direction = transparencyRayDirection;

                        vec3 transparencyColor = trace(transparencyRay, depth + 1);

                        finalColor = (1.0f - mat.transparency) * intersectionColor + mat.transparency * transparencyColor;
                    }
                }

                vec3 lightDir = normalize(-dirLight.direction);
                float diffuseIntensity = glm::max(dot(normal, lightDir), 0.0f);
                vec3 diffuse = sphere.color * diffuseIntensity * dirLight.color;

                vec3 viewDir = normalize(-ray.direction);
                vec3 reflectDir = reflect(-lightDir, normal);
                float specIntensity = pow(glm::max(dot(viewDir, reflectDir), 0.0f), mat.shininess);
                vec3 specular = mat.specular * specIntensity;
                vec3 ambient = mat.ambient;

                vec3 shadowColor = vec3(0.0f);
                int samples = 1;
                vec3 shadowIntensity = vec3(0.0f);
                for (int s = 1; s < samples; ++s) {
                    Ray shadowRay;
                    shadowRay.origin = intersectionPoint;
                    vec3 randomPointOnLight = generateRandomPointOnAreaLight();

                    shadowRay.direction = normalize(randomPointOnLight - intersectionPoint);

                    if (isInShadow(shadowRay)) {
                        // Point is in shadow, accumulate shadow color
                        shadowColor += shadowIntensity;
                    }
                }

                // Average
                shadowColor /= samples;
                finalColor = (ambient + diffuse + specular) * (1.0f - shadowColor);
                return glm::clamp(finalColor, glm::vec3(0.0f), glm::vec3(1.0f));
            }

        }
        for (const Triangle& triangle : triangles) {
            float t;
            if (intersectTriangle(ray, triangle, t) && t < closestT) {
                Material mat = triangle.material;
                closestT = t;
                vec3 intersectionPoint = ray.origin + t * ray.direction;
                vec3 normal = normalize(cross(triangle.v1 - triangle.v0, triangle.v2 - triangle.v0));

                vec3 lightDir = normalize(-dirLight.direction);
                float diffuseIntensity = glm::max(dot(normal, lightDir), 0.0f);
                vec3 diffuse = triangle.color * diffuseIntensity * dirLight.color;
                vec3 viewDir = normalize(-ray.direction);

                vec3 reflectDir = reflect(-lightDir, normal);
                float specIntensity = pow(glm::max(dot(viewDir, reflectDir), 0.0f), mat.shininess);
                vec3 specular = mat.specular * specIntensity;

                vec3 ambient = mat.ambient;
                vec3 shadowColor = vec3(0.0f);
                int numShadowSamples = 1;
                vec3 shadowIntensity = vec3(0.0f);

                for (int shadowSample = 1; shadowSample < numShadowSamples; ++shadowSample) {
                    Ray shadowRay;
                    shadowRay.origin = intersectionPoint;

                    vec3 randomPointOnLight = generateRandomPointOnAreaLight();

                    shadowRay.direction = normalize(randomPointOnLight - intersectionPoint);

                    float shadowT;
                    if (isInShadow(shadowRay)) {
                        // Point is in shadow, accumulate shadow color
                        shadowIntensity += vec3(1.0f);
                    }
                }
                shadowIntensity /= numShadowSamples;
                finalColor = (ambient + diffuse + specular) * (1.0f - shadowIntensity);

                return glm::clamp(finalColor, glm::vec3(0.0f), glm::vec3(1.0f));
            }
        }
        for (const Plane& plane : planes) {
            float t;
            if (intersectPlane(ray, plane, t) && t < closestT) {
                Material mat = plane.material;
                closestT = t;
                vec3 intersectionPoint = ray.origin + t * ray.direction;
                vec3 normal = plane.normal;

                vec3 lightDir = normalize(-dirLight.direction);
                float diffuseIntensity = glm::max(dot(normal, lightDir), 0.0f);
                vec3 diffuse = plane.color * diffuseIntensity * dirLight.color;

                vec3 viewDir = normalize(-ray.direction);
                vec3 reflectDir = reflect(-lightDir, normal);
                float specIntensity = pow(glm::max(dot(viewDir, reflectDir), 0.0f), mat.shininess);
                vec3 specular = mat.specular * specIntensity;

                vec3 ambient = mat.ambient;

                intersectionColor = ambient + diffuse + specular;
                intersectionColor = glm::clamp(intersectionColor, glm::vec3(0.0f), glm::vec3(1.0f));
            }
        }

        return intersectionColor;
    }

    bool isInShadow(const Ray& shadowRay) {
        // Check if the shadow ray intersects with any objects
        for (const Sphere& sphere : spheres) {
            float t;
            if (intersectSphere(shadowRay, sphere, t) && t > 0.0f && t < 1.0f) {
                return true;
            }
        }
        return false;
    }


    bool intersectSphere(const Ray& ray, const Sphere& sphere, float& t) {
        vec3 oc = ray.origin - sphere.center;
        float a = dot(ray.direction, ray.direction);
        float b = 2.0f * dot(oc, ray.direction);
        float c = dot(oc, oc) - sphere.radius * sphere.radius;

        float discriminant = b * b - 4 * a * c;

        if (discriminant > 0) {
            float t1 = (-b - sqrt(discriminant)) / (2.0f * a);
            float t2 = (-b + sqrt(discriminant)) / (2.0f * a);
            t = (t1 < t2) ? t1 : t2;
            return true;
        }
        else if (discriminant == 0) {
            t = -b / (2.0f * a);
            return true;
        }

        return false;
    }

    bool intersectTriangle(const Ray& ray, const Triangle& triangle, float& t) {
        const float EPSILON = 1e-6f;
        vec3 edge1 = triangle.v1 - triangle.v0;
        vec3 edge2 = triangle.v2 - triangle.v0;
        vec3 h = cross(ray.direction, edge2);
        float a = dot(edge1, h);

        if (a > -EPSILON && a < EPSILON) {
            return false;
        }

        float f = 1.0f / a;
        vec3 s = ray.origin - triangle.v0;
        float u = f * dot(s, h);

        if (u < 0.0f || u > 1.0f) {
            return false;
        }

        vec3 q = cross(s, edge1);
        float v = f * dot(ray.direction, q);

        if (v < 0.0f || u + v > 1.0f) {
            return false;
        }

        t = f * dot(edge2, q);

        if (t > EPSILON) {
            return true;
        }

        return false;
    }

    bool intersectPlane(const Ray& ray, const Plane& plane, float& t) {
        float denom = dot(plane.normal, ray.direction);

        if (denom < 1e-6f) {
            vec3 p0 = plane.normal * plane.distance;
            t = dot(p0 - ray.origin, plane.normal) / denom;

            if (t >= 0) {
                return true;
            }
        }

        return false;
    }
};

void parseScene(const char* xmlFileName, Scene& scene) {
    tinyxml2::XMLDocument doc;
    if (doc.LoadFile(xmlFileName) != tinyxml2::XML_SUCCESS) {
        cerr << "Error loading XML file" << endl;
        return;
    }

    tinyxml2::XMLElement* root = doc.FirstChildElement("scene");
    if (!root) {
        cerr << "XML format error: 'scene' element not found." << endl;
        return;
    }

    tinyxml2::XMLElement* lightElem = root->FirstChildElement("lightsource");
    if (lightElem) {
        DirectionalLight dirLight;
        tinyxml2::XMLElement* dirElem = lightElem->FirstChildElement("direction");
        tinyxml2::XMLElement* colorElem = lightElem->FirstChildElement("color");

        if (dirElem && colorElem) {
            dirLight.direction = vec3(dirElem->FloatAttribute("x"), dirElem->FloatAttribute("y"), dirElem->FloatAttribute("z"));
            dirLight.color = vec3(colorElem->FloatAttribute("r"), colorElem->FloatAttribute("g"), colorElem->FloatAttribute("b"));
            scene.setDirectionalLight(dirLight);
        }
    }

    for (tinyxml2::XMLElement* element = root->FirstChildElement(); element; element = element->NextSiblingElement()) {
        const char* type = element->Name();
        if (strcmp(type, "sphere") == 0) {
            Sphere sphere;
            Material material;
            tinyxml2::XMLElement* centerElem = element->FirstChildElement("center");
            tinyxml2::XMLElement* radiusElem = element->FirstChildElement("radius");
            tinyxml2::XMLElement* colorElem = element->FirstChildElement("color");
            tinyxml2::XMLElement* materialElem = element->FirstChildElement("material");

            if (centerElem && radiusElem && colorElem) {
                sphere.center = vec3(centerElem->FloatAttribute("x"), centerElem->FloatAttribute("y"), centerElem->FloatAttribute("z"));
                sphere.radius = radiusElem->FloatAttribute("value");
                sphere.color = vec3(colorElem->FloatAttribute("r"), colorElem->FloatAttribute("g"), colorElem->FloatAttribute("b"));
            }
            if (materialElem) {
                tinyxml2::XMLElement* ambientElem = materialElem->FirstChildElement("ambient");
                tinyxml2::XMLElement* specularElem = materialElem->FirstChildElement("specular");
                tinyxml2::XMLElement* shininessElem = materialElem->FirstChildElement("shininess");
                tinyxml2::XMLElement* transparencyElem = materialElem->FirstChildElement("transparency");

                if (ambientElem) {
                    material.ambient = vec3(ambientElem->FloatAttribute("r"), ambientElem->FloatAttribute("g"), ambientElem->FloatAttribute("b"));
                }
                if (specularElem) {
                    material.specular = vec3(specularElem->FloatAttribute("r"), specularElem->FloatAttribute("g"), specularElem->FloatAttribute("b"));
                }
                if (shininessElem) {
                    material.shininess = shininessElem->FloatAttribute("value");
                }
                if (transparencyElem) {
                    material.transparency = transparencyElem->FloatAttribute("value");
                }
                else {
                    material.transparency = 0.0f; // Default to fully opaque
                }
                sphere.material = material;
            }
            scene.addSphere(sphere);
        }
        else if (strcmp(type, "triangle") == 0) {
            Triangle triangle;
            Material material;
            tinyxml2::XMLElement* v0Elem = element->FirstChildElement("v0");
            tinyxml2::XMLElement* v1Elem = element->FirstChildElement("v1");
            tinyxml2::XMLElement* v2Elem = element->FirstChildElement("v2");
            tinyxml2::XMLElement* colorElem = element->FirstChildElement("color");
            tinyxml2::XMLElement* materialElem = element->FirstChildElement("material");

            if (v0Elem && v1Elem && v2Elem && colorElem) {
                triangle.v0 = vec3(v0Elem->FloatAttribute("x"), v0Elem->FloatAttribute("y"), v0Elem->FloatAttribute("z"));
                triangle.v1 = vec3(v1Elem->FloatAttribute("x"), v1Elem->FloatAttribute("y"), v1Elem->FloatAttribute("z"));
                triangle.v2 = vec3(v2Elem->FloatAttribute("x"), v2Elem->FloatAttribute("y"), v2Elem->FloatAttribute("z"));
                triangle.color = vec3(colorElem->FloatAttribute("r"), colorElem->FloatAttribute("g"), colorElem->FloatAttribute("b"));
            }
            if (materialElem) {
                tinyxml2::XMLElement* ambientElem = materialElem->FirstChildElement("ambient");
                tinyxml2::XMLElement* specularElem = materialElem->FirstChildElement("specular");
                tinyxml2::XMLElement* shininessElem = materialElem->FirstChildElement("shininess");
                tinyxml2::XMLElement* transparencyElem = materialElem->FirstChildElement("transparency");
                if (ambientElem) {
                    material.ambient = vec3(ambientElem->FloatAttribute("r"), ambientElem->FloatAttribute("g"), ambientElem->FloatAttribute("b"));
                }
                if (specularElem) {
                    material.specular = vec3(specularElem->FloatAttribute("r"), specularElem->FloatAttribute("g"), specularElem->FloatAttribute("b"));
                }
                if (shininessElem) {
                    material.shininess = shininessElem->FloatAttribute("value");
                }
                if (transparencyElem) {
                    material.transparency = transparencyElem->FloatAttribute("value");
                }
                else {
                    material.transparency = 0.0f; // Default to fully opaque
                }
                triangle.material = material;
            }
            scene.addTriangle(triangle);
        }
        else if (strcmp(type, "plane") == 0) {
            Plane plane;
            Material material;
            tinyxml2::XMLElement* normalElem = element->FirstChildElement("normal");
            tinyxml2::XMLElement* distanceElem = element->FirstChildElement("distance");
            tinyxml2::XMLElement* colorElem = element->FirstChildElement("color");
            tinyxml2::XMLElement* materialElem = element->FirstChildElement("material");

            if (normalElem && distanceElem && colorElem) {
                plane.normal = vec3(normalElem->FloatAttribute("x"), normalElem->FloatAttribute("y"), normalElem->FloatAttribute("z"));
                plane.distance = distanceElem->FloatAttribute("value");
                plane.color = vec3(colorElem->FloatAttribute("r"), colorElem->FloatAttribute("g"), colorElem->FloatAttribute("b"));

            }
            if (materialElem) {
                tinyxml2::XMLElement* ambientElem = materialElem->FirstChildElement("ambient");
                tinyxml2::XMLElement* specularElem = materialElem->FirstChildElement("specular");
                tinyxml2::XMLElement* shininessElem = materialElem->FirstChildElement("shininess");
                tinyxml2::XMLElement* transparencyElem = materialElem->FirstChildElement("transparency");
                if (ambientElem) {
                    material.ambient = vec3(ambientElem->FloatAttribute("r"), ambientElem->FloatAttribute("g"), ambientElem->FloatAttribute("b"));
                }
                if (specularElem) {
                    material.specular = vec3(specularElem->FloatAttribute("r"), specularElem->FloatAttribute("g"), specularElem->FloatAttribute("b"));
                }
                if (shininessElem) {
                    material.shininess = shininessElem->FloatAttribute("value");
                }
                if (transparencyElem) {
                    material.transparency = transparencyElem->FloatAttribute("value");
                }
                else {
                    material.transparency = 0.0f; // Default to fully opaque
                }
                plane.material = material;
            }
            scene.addPlane(plane);
        }
    }
}
int main() {
    float vehicleTX = 0.0f;
    float vehicleTY = 0.0f;
    float vehicleTZ = 0.0f;
    float vehicleRX = 0.0f;
    float vehicleRY = 0.0f;
    float vehicleRZ = 0.0f;
    int width = 1200;
    int height = 1200;
    Scene scene(width, height);
    parseScene("scene.xml", scene);
    scene.applyTransformations(vehicleTX, vehicleTY, vehicleTZ, vehicleRX, vehicleRY, vehicleRZ);

    CImgDisplay disp(scene.getImage(), "Rendered Image", 0);

    while (!disp.is_closed()) {
        if (disp.is_keyI()) {
            vehicleRX += 1.0f; // Rotate upwards
        }
        if (disp.is_keyK()) {
            vehicleRX -= 1.0f; // Rotate downwards
        }
        if (disp.is_keyL()) {
            vehicleRY += 1.0f; // Rotate right
        }
        if (disp.is_keyJ()) {
            vehicleRY -= 1.0f; // Rotate left
        }
        if (disp.is_keyD()) {
            vehicleTX += 1.0f; // Move Right
        }
        if (disp.is_keyA()) {
            vehicleTX -= 1.0f; // Move left
        }
        if (disp.is_keyW()) {
            vehicleTY += 1.0f; // Move up
        }
        if (disp.is_keyS()) {
            vehicleTY -= 1.0f; // Move down
        }
        // Update positions and rotations of all objects 
        scene.applyTransformations(vehicleTX, vehicleTY, vehicleTZ, vehicleRX, vehicleRZ, vehicleRZ);
        scene.render();
        const CImg<unsigned char>& updatedImage = scene.getImage();
        disp.display(updatedImage);

        disp.wait();
    }

    return 0;
}