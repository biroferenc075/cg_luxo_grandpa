//=============================================================================================
// Mintaprogram: Zöld háromszög. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!!
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Biró Ferenc
// Neptun : HR4VCG
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"

const char *vertexSource = R"(
	#version 330
    precision highp float;

	uniform vec3 wLookAt, wRight, wUp;

	layout(location = 0) in vec2 cCamWindowVertex;
	out vec3 p;

	void main() {
		gl_Position = vec4(cCamWindowVertex, 0, 1);
		p = wLookAt + wRight * cCamWindowVertex.x + wUp * cCamWindowVertex.y;
	}
)";
const char *fragmentSource = R"(
	#version 330
    precision highp float;

	const float M_PI = 3.141592654;

	vec4 qmul(vec4 q1, vec4 q2) {
		vec3 d1 = vec3(q1.x, q1.y, q1.z), d2 = vec3(q2.x, q2.y, q2.z);
		return vec4(d2 * q1.w + d1 * q2.w + cross(d1, d2),
		q1.w * q2.w - dot(d1, d2));
	}
	vec4 quat(float ang, vec3 axis) {
		vec3 d = normalize(axis) * sin(ang / 2);
		return vec4(d.x, d.y, d.z, cos(ang / 2));
	}
	vec3 Rotate(vec3 u, vec4 q) {
		vec4 qinv = vec4(-q.x, -q.y, -q.z, q.w);
		vec4 qr = qmul(qmul(q, vec4(u.x, u.y, u.z, 0)), qinv);
		return vec3(qr.x, qr.y, qr.z);
	}
	struct Material {
		vec3 ka, kd, ks;
		float  shininess;
	};
	struct Light {
		vec3 pos;
		vec3 Le, La;
	};
	struct Sphere {
		vec3 center;
		float radius;
	};
    struct Cylinder {
        vec3 base;
		vec3 dir;
        float radius;
        float height;
    };
    struct Paraboloid {
        vec3 base;
        vec3 dir;
        float height;
		float scale;
    };
    struct Plane {
        vec3 pos;
        vec3 normal;
    };
	struct Circle {
		vec3 pos;
		vec3 normal;
		float rad;
	};
	struct Ray {
		vec3 start, dir;
	};
    struct Hit {
        float t;
        vec3 position, normal;
        int mat;
    };
	Sphere Rotate(Sphere s, vec4 q) {
		Sphere res = Sphere(Rotate(s.center,q),s.radius);
		return res;
	}
	Cylinder Rotate(Cylinder c, vec4 q) {
		Cylinder res = Cylinder( Rotate(c.base,q),  Rotate(c.dir,q),c.radius,c.height);
		return res;
	}
	Paraboloid Rotate(Paraboloid p, vec4 q) {
		Paraboloid res = Paraboloid(Rotate(p.base,q),  Rotate(p.dir,q),p.height,p.scale);
		return res;
	}
	Ray Rotate(Ray r, vec4 q) {
		Ray res = Ray(Rotate(r.start,q), Rotate(r.dir,q));
		return res;
	}
	Light Rotate(Light l, vec4 q) {
		Light res = Light(Rotate(l.pos,q),l.Le,l.La);
		return res;
	}
	Sphere Rotate(Sphere s, vec4 q, vec3 pivot) {
		Sphere res = Sphere(Rotate(s.center-pivot,q)+pivot,s.radius);
		return res;
	}
	Cylinder Rotate(Cylinder c, vec4 q, vec3 pivot) {
		Cylinder res = Cylinder( Rotate(c.base-pivot,q)+pivot, Rotate(c.dir,q),c.radius,c.height);
		return res;
	}
	Paraboloid Rotate(Paraboloid p, vec4 q, vec3 pivot) {
		Paraboloid res = Paraboloid(Rotate(p.base-pivot,q)+pivot, Rotate(p.dir,q),p.height,p.scale);
		return res;
	}
	Ray Rotate(Ray r, vec4 q, vec3 pivot) {
		Ray res = Ray(Rotate(r.start-pivot,q)+pivot, Rotate(r.dir,q));
		return res;
	}
	Light Rotate(Light l, vec4 q, vec3 pivot) {
		Light res = Light(Rotate(l.pos-pivot,q)+pivot,l.Le,l.La);
		return res;
	}
	uniform Material materials[3];
    uniform Plane ground;
    uniform Cylinder base;
    uniform Circle basePlate;
    uniform Sphere joint1;
    uniform Sphere joint2;
    uniform Sphere joint3;

    uniform Cylinder arm1;
    uniform Cylinder arm2;

    uniform Paraboloid head;
    uniform Light bulb;
    uniform Light roomLight;
    uniform vec3 wEye;

	in  vec3 p;
	out vec4 fragmentColor;

	Hit intersect(const Sphere object, const Ray ray) {
		Hit hit;
		hit.t = -1;
		vec3 dist = ray.start - object.center;
		float a = dot(ray.dir, ray.dir);
		float b = dot(dist, ray.dir) * 2.0f;
		float c = dot(dist, dist) - object.radius * object.radius;
		float discr = b * b - 4.0f * a * c;

		if (discr < 0)	return hit;
		float sqrt_discr = sqrt(discr);
		float t1 = (-b + sqrt_discr) / 2.0f / a;
		float t2 = (-b - sqrt_discr) / 2.0f / a;

		if (t1 <= 0) return hit;

		hit.t = (t2 > 0) ? t2 : t1;
		hit.position = ray.start + ray.dir * hit.t;
		hit.normal = (hit.position - object.center) / object.radius;

		hit.mat = 2;
		return hit;
	}
	Hit intersect(const Cylinder object, const Ray r) {
		vec3 halfway = normalize(vec3(0,1,0) + object.dir);
		Cylinder cyl =  Rotate(object, quat(M_PI, halfway));
		Ray ray = Rotate(r, quat(M_PI, halfway));
		Hit hit;
		hit.t = -1;
		vec2 dist = ray.start.xz - cyl.base.xz;
		float a = dot(ray.dir.xz, ray.dir.xz);
		float b = dot(dist, ray.dir.xz) * 2.0f;
		float c = dot(dist, dist) - cyl.radius * cyl.radius;
		float discr = b * b - 4.0f * a * c;

		if (discr < 0) return hit;

		float sqrt_discr = sqrt(discr);
		float t1 = (-b + sqrt_discr) / 2.0f / a;
		float t2 = (-b - sqrt_discr) / 2.0f / a;

		if (t1 <= 0) return hit;

		hit.t = (t2 > 0) ? t2 : t1;
		hit.position = ray.start + ray.dir * hit.t;
		if(hit.position.y < cyl.base.y || hit.position.y > cyl.base.y+cyl.height) {
			hit.t = -1;
			return hit;
		}
		vec3 normal = (hit.position - cyl.base);
		normal.y = 0;
		normal/=cyl.radius;
		hit.position = Rotate(hit.position,quat(M_PI, halfway)) ;
		hit.normal = Rotate(normal,quat(M_PI, halfway));

		hit.mat = 1;
		return hit;
	}
    Hit intersect(const Plane plane, const Ray ray) {
        Hit hit;
        float ang = dot(ray.dir,plane.normal);

		float t = dot(plane.pos-ray.start,plane.normal)/ang;
        if(t < 0) return hit;

        hit.t = t;
        hit.position = ray.start + ray.dir * hit.t;
        hit.normal = plane.normal;
		hit.mat = 0;
        return hit;
    }
	Hit intersect(const Circle circle, const Ray ray) {
		Hit hit;
		float ang = dot(ray.dir,circle.normal);

		float t = dot(circle.pos-ray.start,circle.normal)/ang;
		if(t < 0) return hit;
		hit.t = t;
		hit.position = ray.start + ray.dir * hit.t;
		vec3 dist = hit.position-circle.pos;
		if(length(dist) > circle.rad) hit.t = -1;

		hit.normal = circle.normal;
		hit.mat = 1;
		return hit;
	}
	Hit intersect(const Paraboloid object, const Ray r) {
		vec3 halfway = normalize(vec3(0,1,0) + object.dir);
		if(object.dir == vec3(0,-1,0)) halfway = vec3(1,0,0);
		Paraboloid par = Rotate(object, quat(M_PI, halfway));
		Ray ray = Rotate(r, quat(M_PI, halfway));

		Hit hit;
		hit.t = -1;
		vec3 dist = ray.start - par.base;
		float a = dot(ray.dir.xz, ray.dir.xz)/par.scale;
		float b = dot(dist.xz, ray.dir.xz) * 2.0f/par.scale-ray.dir.y;
		float c = dot(dist.xz, dist.xz)/par.scale-dist.y;
		float discr = b * b - 4.0f * a * c;

		if (discr < 0) return hit;

		float sqrt_discr = sqrt(discr);
		float t1 = (-b + sqrt_discr) / 2.0f / a;
		float t2 = (-b - sqrt_discr) / 2.0f / a;

		if(t2 > 0) {
			vec3 delta2 = ray.start + ray.dir * t2 - par.base;
			if (delta2.y < 0 || delta2.y > par.height) {
				t2 = -1;
			}
		}
		if(t1 > 0) {
			vec3 delta1 = ray.start + ray.dir * t1 - par.base;
			if (delta1.y < 0 || delta1.y > par.height) {
				t1 = -1;
			}
		}
		if (t1 <= 0 && t2 <= 0)  {
			hit.t = -1;
			return hit;
		}
		else hit.t = (t2 > 0) ? t2 : t1;
		hit.position = ray.start + ray.dir * hit.t;
		vec3 delta = hit.position - par.base;

		vec3 normal = normalize(vec3(delta.x*2/par.scale, -1, delta.z*2/par.scale));
		hit.position = Rotate(hit.position,quat(M_PI, halfway));
		hit.normal = Rotate(normal,quat(M_PI, halfway));
		hit.mat = 1;
		return hit;
	}
	Hit betterHit(Hit hit1, Hit hit2) {
		if (hit2.t > 0 && (hit1.t < 0 || hit2.t < hit1.t)) return hit2;
		return hit1;
    }
	Hit firstIntersect(Ray ray) {
		Hit bestHit;
		bestHit.t = -1;
		Hit hit;
		hit = intersect(ground, ray);
		bestHit = betterHit(bestHit, hit);
		hit = intersect(joint1, ray);
		bestHit = betterHit(bestHit, hit);
		hit = intersect(joint2, ray);
		bestHit = betterHit(bestHit, hit);
		hit = intersect(joint3, ray);
		bestHit = betterHit(bestHit, hit);
		hit = intersect(base, ray);
		bestHit = betterHit(bestHit, hit);
		hit = intersect(basePlate, ray);
		bestHit = betterHit(bestHit, hit);
		hit = intersect(arm1, ray);
		bestHit = betterHit(bestHit, hit);
		hit = intersect(arm2, ray);
		bestHit = betterHit(bestHit, hit);
		hit = intersect(head, ray);
		bestHit = betterHit(bestHit, hit);

		if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
		return bestHit;
	}

	bool shadowIntersect(Ray ray, float lightDist) {
		float t;
		t = intersect(head, ray).t;
		if(t > 0 && t < lightDist) return true;
		t = intersect(ground, ray).t;
		if(t > 0 && t < lightDist) return true;
		t = intersect(base, ray).t;
		if(t > 0 && t < lightDist) return true;
		t = intersect(basePlate, ray).t;
		if(t > 0 && t < lightDist) return true;
		t = intersect(joint1, ray).t;
		if(t > 0 && t < lightDist) return true;
		t = intersect(joint2, ray).t;
		if(t > 0 && t < lightDist) return true;
		t = intersect(joint3, ray).t;
		if(t > 0 && t < lightDist) return true;
		t = intersect(arm1, ray).t;
		if(t > 0 && t < lightDist) return true;
		t = intersect(arm2, ray).t;
		if(t > 0 && t < lightDist) return true;
		return false;
	}

	const float epsilon = 0.01f;

	vec3 calcRadiance(Light l, Ray ray, Hit hit) {
		if (hit.t < 0) return l.La;
		vec3 outRadiance = vec3(0,0,0);
		outRadiance += materials[hit.mat].ka * l.La;
		Ray shadowRay;
		shadowRay.start = hit.position + hit.normal * epsilon;
		shadowRay.dir = normalize(l.pos-shadowRay.start);
		float lightDist = dot(l.pos-shadowRay.start,l.pos-shadowRay.start);
		float cosTheta = dot(hit.normal, shadowRay.dir);
		if (cosTheta > 0 && !shadowIntersect(shadowRay, sqrt(lightDist))) {
			vec3 luminosity = l.Le/lightDist;
			outRadiance += 50 * luminosity/lightDist * materials[hit.mat].kd * cosTheta;
			vec3 halfway = normalize(-ray.dir + shadowRay.dir);
			float cosDelta = dot(hit.normal, halfway);
			if (cosDelta > 0) outRadiance += 100*luminosity/lightDist * materials[hit.mat].ks * pow(cosDelta, materials[hit.mat].shininess);
		}
		return outRadiance;
	}
	vec3 trace(Ray ray) {
		vec3 outRadiance = vec3(0, 0, 0);

		Hit hit = firstIntersect(ray);
		outRadiance += calcRadiance(roomLight, ray, hit);
		outRadiance += calcRadiance(bulb, ray, hit);
		return outRadiance;
	}
	void main() {
		Ray ray;
		ray.start = wEye;
		ray.dir = normalize(p - wEye);
		fragmentColor = vec4(trace(ray), 1);
	}
)";
struct Material {
    vec3 ka, kd, ks;
    float  shininess;
    Material(vec3 _ka, vec3 _kd, vec3 _ks, float _shininess) {
        ka = _ka;
        kd = _kd;
        ks = _ks;
        shininess = _shininess;
    }
};

vec4 qmul(vec4 q1, vec4 q2) {
    vec3 d1 = vec3(q1.x, q1.y, q1.z), d2 = vec3(q2.x, q2.y, q2.z);
    vec3 r = d2 * q1.w + d1 * q2.w + cross(d1, d2);
    return vec4(r.x,r.y,r.z,q1.w * q2.w - dot(d1, d2));
}
vec4 quat(float ang, vec3 axis) {
    vec3 d = normalize(axis) * sin(ang / 2);
    return vec4(d.x, d.y, d.z, cos(ang / 2));
}
vec3 Rotate(vec3 u, vec4 q) {
    vec4 qinv = vec4(-q.x, -q.y, -q.z, q.w);
    vec4 qr = qmul(qmul(q, vec4(u.x, u.y, u.z, 0)), qinv);
    return vec3(qr.x, qr.y, qr.z);
}
struct Light {
    vec3 pos;
    vec3 Le, La;
    Light(vec3 _pos, vec3 _Le, vec3 _La) {
        pos = _pos;
        Le = _Le;
        La = _La;
    };
    Light() {
        pos = vec3(0,0,0);
        Le = vec3(0,0,0);
        La = vec3(0,0,0);
    }
};
struct Sphere {
    vec3 center;
    float radius;
    Sphere(vec3 _center, float _radius) {
        center = _center;
        radius = _radius;
    }
    Sphere() {
        center = vec3(0,0,0);
        radius = 0;
    }
};
struct Cylinder {
    vec3 base;
    vec3 dir;
    float radius;
    float height;
    Cylinder(vec3 _base, vec3 _dir, float _radius, float _height) {
        base = _base;
        dir = _dir;
        radius = _radius;
        height = _height;
    }
    Cylinder() {
        base = vec3(0,0,0);
        dir = vec3(0,1,0);
        radius = 0;
        height = 0;
    }
};
struct Paraboloid {
    vec3 base;
    vec3 dir;
    float height;
    float scale;
    Paraboloid(vec3 _base, vec3 _dir, float _height, float _scale) {
        base = _base;
        dir = _dir;
        height = _height;
        scale = _scale;
    }
    Paraboloid() {
        base = vec3(0,0,0);
        dir = vec3(0,1,0);
        height = 0;
        scale = 0;
    }
};
struct Plane {
    vec3 pos;
    vec3 normal;
    Plane(vec3 _pos, vec3 _normal) {
        pos = _pos;
        normal = _normal;
    }
};
struct Circle {
    vec3 pos;
    vec3 normal;
    float rad;
    Circle(vec3 _pos, vec3 _normal, float _rad) {
        pos = _pos;
        normal = _normal;
        rad = _rad;
    }
    Circle() {
        pos = vec3(0,0,0);
        normal = vec3(0,1,0);
        rad = 0;
    }
};
Sphere Rotate(Sphere s, vec4 q, vec3 pivot) {
    Sphere res = Sphere(Rotate(s.center-pivot,q)+pivot,s.radius);
    return res;
}
Cylinder Rotate(Cylinder c, vec4 q, vec3 pivot) {
    Cylinder res = Cylinder( Rotate(c.base-pivot,q)+pivot,  Rotate(c.dir,q),c.radius,c.height);
    return res;
}
Paraboloid Rotate(Paraboloid p, vec4 q, vec3 pivot) {
    Paraboloid res = Paraboloid(Rotate(p.base-pivot,q)+pivot,  Rotate(p.dir,q),p.height,p.scale);
    return res;
}
Light Rotate(Light l, vec4 q, vec3 pivot) {
    Light res = Light(Rotate(l.pos-pivot,q)+pivot,l.Le,l.La);
    return res;
}
const float baseRad = 3.5;
const float baseHeight = 1.0;
const float jointRad = 0.45;
const float jointDelta = 0.2;
const float armLen = 8.0;
const float armRad = 0.25;
const float headHeight = 4;
const float headSize = 1.0;
const vec3 eyeDelta = vec3(10,-10,0);

struct Camera {
    vec3 eye, lookat, right, up;
    float fov;
public:
    void set(vec3 _eye, vec3 _lookat, vec3 vup, float _fov) {
        eye = _eye;
        lookat = _lookat;
        fov = _fov;
        vec3 w = eye - lookat;
        float f = length(w);
        right = normalize(cross(vup, w)) * f * tanf(fov / 2);
        up = normalize(cross(w, right)) * f * tanf(fov / 2);
    }
    void Animate(float dt) {
        eye = vec3((eye.x - lookat.x) * cos(dt) + (eye.z - lookat.z) * sin(dt) + lookat.x,
                   eye.y,
                   -(eye.x - lookat.x) * sin(dt) + (eye.z - lookat.z) * cos(dt) + lookat.z);
        set(eye, lookat, up, fov);
    }
};

class Shader : public GPUProgram {
public:
    void setUniformCamera(const Camera& camera) {
        setUniform(camera.eye,"wEye");
        setUniform(camera.lookat,"wLookAt");
        setUniform(camera.right,"wRight");
        setUniform(camera.up,"wUp");
    }
};

class Scene {
    Camera camera;
    Cylinder base;
    Circle basePlate;
    Sphere joint1;
    Sphere joint2;
    Sphere joint3;
    std::vector<Material> materials;
    Cylinder arm1;
    Cylinder arm2;
    Paraboloid head;
    Light bulb;
    Plane ground = Plane(vec3(0,0,0)+eyeDelta,vec3(0,1,0));
    Light roomLight;
public:
    Scene() {
        vec3 armdir1 = normalize(vec3(1,1,0));
        vec3 armdir2 = normalize(vec3(1,1,2));
        vec3 headdir = normalize(vec3(-0.5,-0.5,20));
        base = Cylinder(vec3(0,0,0)+eyeDelta,vec3(0,1,0),baseRad,baseHeight);
        basePlate = Circle(vec3(0,1,0)+eyeDelta,vec3(0,1,0),baseRad);
        joint1 = Sphere(basePlate.pos,jointRad);
        arm1 = Cylinder(joint1.center+armdir1*jointDelta,armdir1,armRad,armLen);
        joint2 = Sphere(arm1.base+arm1.dir*(armLen+jointDelta),jointRad);
        arm2 = Cylinder(joint2.center+armdir2*jointDelta,armdir2,armRad,armLen);
        joint3 = Sphere(arm2.base+arm2.dir*(armLen+jointDelta),jointRad);
        head = Paraboloid(joint3.center+(jointRad)*headdir,headdir,headHeight,headSize);
        roomLight = Light(vec3(16,10,-10),vec3(4000,4000,2000),vec3(0.15f,0.15f,0.15f));
        bulb = Light(head.base+head.dir*(1/(4*head.scale)),vec3(300,250,250),vec3(0.1f,0.1f,0.05f));

        vec3 kd1 = vec3(0.350, 0.2, 0.160);
        vec3 kd2 = vec3(0.392, 0.572, 0.890);
        vec3 kd3 = vec3(0.933, 0.933, 0.588);
        materials = std::vector<Material>();
        materials.push_back(Material(kd1*M_PI,kd1, vec3(1.0, 0.8, 0.8),300));
        materials.push_back(Material(kd2*M_PI,kd2, vec3(0.3, 0.3, 1.5),120));
        materials.push_back(Material(kd3*M_PI,kd3, vec3(1, 1, 0.1), 20));
    }
    void build() {
        vec3 eye = vec3(50, 0, 0);
        vec3 vup = vec3(0, 1, 0);
        vec3 lookat = vec3(0, 0, 0);
        float fov = 45 * (float)M_PI / 180;
        camera.set(eye, lookat, vup, fov);
    }
    void SetUniform(Shader& sh) {
        sh.setUniform(materials.at(0).ka,"materials[0].ka");
        sh.setUniform(materials.at(0).ks,"materials[0].ks");
        sh.setUniform(materials.at(0).kd,"materials[0].kd");
        sh.setUniform(materials.at(0).shininess,"materials[0].shininess");

        sh.setUniform(materials.at(1).ka,"materials[1].ka");
        sh.setUniform(materials.at(1).ks,"materials[1].ks");
        sh.setUniform(materials.at(1).kd,"materials[1].kd");
        sh.setUniform(materials.at(1).shininess,"materials[1].shininess");

        sh.setUniform(materials.at(2).ka,"materials[2].ka");
        sh.setUniform(materials.at(2).ks,"materials[2].ks");
        sh.setUniform(materials.at(2).kd,"materials[2].kd");
        sh.setUniform(materials.at(2).shininess,"materials[2].shininess");

        sh.setUniform(ground.pos,"ground.pos");
        sh.setUniform(ground.normal,"ground.normal");

        sh.setUniform(base.dir,"base.dir");
        sh.setUniform(base.base,"base.base");
        sh.setUniform(base.height,"base.height");
        sh.setUniform(base.radius,"base.radius");

        sh.setUniform(bulb.pos,"bulb.pos");
        sh.setUniform(bulb.Le,"bulb.Le");
        sh.setUniform(bulb.La,"bulb.La");

        sh.setUniform(roomLight.pos,"roomLight.pos");
        sh.setUniform(roomLight.Le,"roomLight.Le");
        sh.setUniform(roomLight.La,"roomLight.La");

        sh.setUniform(basePlate.pos,"basePlate.pos");
        sh.setUniform(basePlate.normal,"basePlate.normal");
        sh.setUniform(basePlate.rad,"basePlate.rad");

        sh.setUniform(arm1.base,"arm1.base");
        sh.setUniform(arm1.dir,"arm1.dir");
        sh.setUniform(arm1.height,"arm1.height");
        sh.setUniform(arm1.radius,"arm1.radius");

        sh.setUniform(arm2.base,"arm2.base");
        sh.setUniform(arm2.dir,"arm2.dir");
        sh.setUniform(arm2.height,"arm2.height");
        sh.setUniform(arm2.radius,"arm2.radius");

        sh.setUniform(joint1.center,"joint1.center");
        sh.setUniform(joint1.radius,"joint1.radius");

        sh.setUniform(joint2.center,"joint2.center");
        sh.setUniform(joint2.radius,"joint2.radius");

        sh.setUniform(joint3.center,"joint3.center");
        sh.setUniform(joint3.radius,"joint3.radius");

        sh.setUniform(head.base,"head.base");
        sh.setUniform(head.dir,"head.dir");
        sh.setUniform(head.height,"head.height");
        sh.setUniform(head.scale,"head.scale");

        sh.setUniformCamera(camera);
    }
    void Animate(float dt) {
        camera.Animate(dt/2.0);
        vec4 q1 = quat(dt/1.5,vec3(1, 3,1));
        vec4 q2 = quat(dt/1.5,vec3(1,1,0));
        vec4 q3 = quat(dt/1.5,vec3(5,1,0));

        arm1 = Rotate(arm1, q1,joint1.center);
        joint2 = Rotate(joint2, q1,joint1.center);
        arm2 = Rotate(arm2, q1,joint1.center);
        joint3 = Rotate(joint3, q1,joint1.center);
        head = Rotate(head, q1,joint1.center);
        bulb = Rotate(bulb, q1, joint1.center);

        arm2 = Rotate(arm2, q2, joint2.center);
        joint3 = Rotate(joint3, q2,joint2.center);
        head = Rotate(head, q2, joint2.center);
        bulb = Rotate(bulb, q2, joint2.center);

        head = Rotate(head, q3, joint3.center);
        bulb = Rotate(bulb, q3, joint3.center);
    }
};

Shader shader;
Scene scene = Scene();

class FullScreenTexturedQuad {
    unsigned int vao = 0;
public:
    void create() {
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);

        unsigned int vbo;
        glGenBuffers(1, &vbo);

        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
    }

    void Draw() {
        glBindVertexArray(vao);
        glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
    }
};

FullScreenTexturedQuad fullScreenTexturedQuad;

void onInitialization() {
    glViewport(0, 0, windowWidth, windowHeight);
    shader.create(vertexSource, fragmentSource, "fragmentColor");
    shader.Use();
    scene.build();
    fullScreenTexturedQuad.create();
}
void onDisplay() {
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    scene.SetUniform(shader);
    fullScreenTexturedQuad.Draw();

    glutSwapBuffers();
}
void onKeyboard(unsigned char key, int pX, int pY) {
}
void onKeyboardUp(unsigned char key, int pX, int pY) {
}
void onMouse(int button, int state, int pX, int pY) {
}
void onMouseMotion(int pX, int pY) {
}
void onIdle() {
    scene.Animate(0.015f);
    glutPostRedisplay();
}