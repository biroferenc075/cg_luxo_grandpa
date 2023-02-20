
	#version 450
    precision highp float;

	uniform int frames;
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
		Cylinder res = Cylinder( Rotate(c.base-pivot,q)+pivot,  Rotate(c.dir,q),c.radius,c.height);
		return res;
	}
	Paraboloid Rotate(Paraboloid p, vec4 q, vec3 pivot) {
		Paraboloid res = Paraboloid(Rotate(p.base-pivot,q)+pivot,  Rotate(p.dir,q),p.height,p.scale);
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
	const float baseRad = 3.5;
	const float baseHeight = 1.0;
	const float jointRad = 0.45;
	const float jointDelta = 0.2;
	const float armLen = 8.0;
	const float armRad = 0.25;
	const float headHeight = 4;
	const float headSize = 1.3;
	const vec3 eyeDelta = vec3(0,-10,0);



	Cylinder base;
	Circle basePlate;

	Sphere joint1;
	Sphere joint2;
	Sphere joint3;

	Cylinder arm1;
	Cylinder arm2;

	Paraboloid head;
	Light bulb;

	const Plane ground = Plane(vec3(0,0,0)+eyeDelta,vec3(0,1,0));

	uniform vec3 wEye;
	Light roomLight;
	Material materials[4];

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
		float t1 = (-b + sqrt_discr) / 2.0f / a;	// t1 >= t2 for sure
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
        if(ang == 0) return hit;

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
		if(ang == 0) return hit;

		float t = dot(circle.pos-ray.start,circle.normal)/ang;
		if(t < 0) return hit;
		hit.t = t;
		hit.position = ray.start + ray.dir * hit.t;
		vec3 dist = hit.position-circle.pos;
		if(dot(dist, dist) > circle.rad*circle.rad) hit.t = -1;

		hit.normal = circle.normal;
		hit.mat = 1;
		return hit;
	}
	Hit intersect(const Paraboloid object, const Ray r) {
		vec3 halfway = normalize(vec3(0,1,0) + object.dir);
		if(object.dir == vec3(0,-1,0)) halfway = vec3(1,0,0);
		Paraboloid par =  Rotate(object, quat(M_PI, halfway));
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
		float t1 = (-b + sqrt_discr) / 2.0f / a;	// t1 >= t2 for sure
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

		vec3 normal = normalize(vec3(delta.x*2, -1, delta.z*2));
		hit.position = Rotate(hit.position,quat(M_PI, halfway)) ;
		hit.normal = Rotate(normal,quat(M_PI, halfway));
		hit.mat = 1;
		if(t2 < 0 && t1 > 0) hit.mat = 3;
		if(dot(hit.normal,r.dir) < 0) hit.normal *= -1;
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

	vec3 calcRadiance(Light l, Ray ray, Hit hit, vec3 weight) {
		if (hit.t < 0) return weight * l.La;
		vec3 outRadiance = vec3(0,0,0);
		outRadiance += weight * materials[hit.mat].ka * l.La;
		Ray shadowRay;
		shadowRay.start = hit.position + hit.normal * epsilon;
		shadowRay.dir = normalize(l.pos-shadowRay.start);
		float lightDist = dot(l.pos-shadowRay.start,l.pos-shadowRay.start);
		float cosTheta = dot(hit.normal, shadowRay.dir);
		if (cosTheta > 0 && !shadowIntersect(shadowRay, sqrt(lightDist))) {
			outRadiance += debug;
			vec3 luminosity = 100*l.Le/lightDist;
			outRadiance += weight * luminosity/lightDist * materials[hit.mat].kd * cosTheta;
			vec3 halfway = normalize(-ray.dir + shadowRay.dir);
			float cosDelta = dot(hit.normal, halfway);
			if (cosDelta > 0) outRadiance += weight * luminosity/lightDist * materials[hit.mat].ks * pow(cosDelta, materials[hit.mat].shininess);
		}
		return outRadiance;
	}
	vec3 trace(Ray ray) {
		vec3 weight = vec3(1, 1, 1);
		vec3 outRadiance = vec3(0, 0, 0);

		Hit hit = firstIntersect(ray);
		outRadiance += calcRadiance(roomLight, ray, hit, weight);
		outRadiance += calcRadiance(bulb, ray, hit, weight);
		return outRadiance;
	}




	void main() {
		float delta = 0.01;
		vec3 armdir1 = normalize(vec3(1,1,0));
		vec3 armdir2 = normalize(vec3(0,1,2));
		vec3 headdir = normalize(vec3(-10,-4,4));

		base = Cylinder(vec3(0,0,0)+eyeDelta,vec3(0,1,0),baseRad,baseHeight);
		basePlate = Circle(vec3(0,1,0)+eyeDelta,vec3(0,1,0),baseRad);
		joint1 = Sphere(basePlate.pos,jointRad);
		arm1 = Cylinder(joint1.center+armdir1*delta,armdir1,armRad,armLen);
		joint2 = Sphere(arm1.base+arm1.dir*armLen,jointRad);
		arm2 = Cylinder(joint2.center+armdir2*delta,armdir2,armRad,armLen);
		joint3 = Sphere(arm2.base+arm2.dir*armLen,jointRad);

		head = Paraboloid(joint3.center+delta*headdir,headdir,headHeight,headSize);

		vec3 kd1 = vec3(0.450, 0.376, 0.360);
		vec3 kd2 = vec3(0.392, 0.572, 0.890);
		vec3 kd3 = vec3(0.933, 0.933, 0.588);
		vec3 kd4 = vec3(4.0,4.0,4.0);

		materials[0] = Material(kd1*M_PI,kd1, vec3(2, 2, 2),2000);
		materials[1] = Material(kd2*M_PI,kd2, vec3(2, 2, 2),2000);
		materials[2] = Material(kd3*M_PI,kd3, vec3(2, 2, 2),2000);
		materials[3] = Material(kd4*M_PI,kd4, vec3(2, 2, 2),2000);

		roomLight = Light(vec3(8,4,5),vec3(800,800,400),vec3(0.15f,0.15f,0.15f));
		bulb = Light(head.base+head.dir*(jointRad+2),vec3(300,250,250),vec3(0.1f,0.1f,0.05f));
		dSphere = Sphere(bulb.pos,0.1);
		float dT = 0.5*frames/60.0;
		vec4 q1 = quat(dT,vec3(2, 4,1));
		vec4 q2 = quat(dT,vec3(4,-5,1));
		vec4 q3 = quat(dT,vec3(-5,6,0));

		arm1 = Rotate(arm1, q1,arm1.base);
		joint2 = Rotate(joint2, q1,arm1.base);
		arm2 = Rotate(arm2, q1,arm1.base);
		joint3 = Rotate(joint3, q1,arm1.base);
		head = Rotate(head, q1,arm1.base);
		bulb = Rotate(bulb, q1, arm1.base);

		arm2 = Rotate(arm2, q2,arm2.base);
		joint3 = Rotate(joint3, q2,arm2.base);
		head = Rotate(head, q2, arm2.base);
		bulb = Rotate(bulb, q2, arm2.base);

		head = Rotate(head, q3, head.base);
		bulb = Rotate(bulb, q3, head.base);

		Ray ray;
		ray.start = wEye;
		ray.dir = normalize(p - wEye);
		fragmentColor = vec4(trace(ray), 1);
	}