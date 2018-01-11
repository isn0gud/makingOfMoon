#include <glm/vec4.hpp>

#ifndef BARNES_HUT_TREE_HPP_
#define BARNES_HUT_TREE_HPP_

//  units: SI, but km instead of m
//  6.674×10−20 (km)^3⋅kg^(−1)⋅s^(−2)
const float G = 6.674E-20;
const float distanceEpsilon = 47.0975;

/*
Implementation of Barnes-Hut algorithm
*/

// Class that represents a cubic cell.
// Can check if a point is within the area.
// Can be used to get an octant subarea of the cube.
class Cube {
private:
	glm::vec4 center;
	float length;

public:
	Cube() : length(0) {}
	Cube(glm::vec4 c, float l) : center(c), length(l) {}

    void setLength(float l) {length = l;}

	glm::vec4 getCenter() { return center; }
	float getLength() { return length; }

	// Check if a position is within the cube
	bool within(glm::vec4 p) {
		return (p.x >= center.x - (length / 2) && p.x <= center.x + (length / 2) &&   // Within x
			    p.y >= center.y - (length / 2) && p.y <= center.y + (length / 2) &&   // Within y
			    p.z >= center.z - (length / 2) && p.z <= center.z + (length / 2));    // Within z
	}

	// Get a subarea of the cube
	Cube getOctant(int i) {
		switch (i) {
		case 0: return Cube(center + glm::vec4(length / 4, length / 4, length / 4, 1), length / 2);	     // top front right
		case 1: return Cube(center + glm::vec4(-length / 4, length / 4, length / 4, 1), length / 2);	 // top front left
		case 2: return Cube(center + glm::vec4(length / 4, -length / 4, length / 4, 1), length / 2);	 // top back right
		case 3: return Cube(center + glm::vec4(length / 4, length / 4, -length / 4, 1), length / 2);	 // bottom front right
		case 4: return Cube(center + glm::vec4(-length / 4, -length / 4, length / 4, 1), length / 2);	 // top back left
		case 5: return Cube(center + glm::vec4(length / 4, -length / 4, -length / 4, 1), length / 2);	 // bottom back right
		case 6: return Cube(center + glm::vec4(-length / 4, length / 4, -length / 4, 1), length / 2);	 // bottom front left
		case 7: return Cube(center + glm::vec4(-length / 4, -length / 4, -length / 4, 1) , length / 2);  // bottom back left
		default: return Cube(center, length);
		}
	}
};

// Represents the total mass and center of gravity for a single or a group of particles
class Body {
private:
	float mass;
	glm::vec4 position;

public:
	Body() : mass(0) {}
	Body(float m, glm::vec4 p) : mass(m), position(p) {}
	glm::vec4 getPosition() { return position; }
	float getMass() { return mass; }
	bool within(Cube c) {
		return c.within(position);
	}
};

// Merge body A and B
// Calculate their new mass and new center of gravity
Body add(Body a, Body b) {
	float mass = a.getMass() + b.getMass();
	glm::vec4 position;
	position.x = ((a.getPosition().x * a.getMass()) + (b.getPosition().x * b.getMass())) / mass;
	position.y = ((a.getPosition().y * a.getMass()) + (b.getPosition().y * b.getMass())) / mass;
	position.z = ((a.getPosition().z * a.getMass()) + (b.getPosition().z * b.getMass())) / mass;
	position.w = 1;
	return Body(mass, position);
}

// Calculate the gravitational force between two bodies
glm::vec3 gravitationalForce(Body a, Body b) {
	glm::vec3 force(0, 0, 0);
	glm::vec3 difference = a.getPosition() - b.getPosition();
	float distance = glm::length(difference);
	glm::vec3 differenceNormal = (difference / distance);

	// Newtonian gravity (F_g = -G m_1 m_2 r^(-2) hat{n}), doubles are needed to prevent overflow, needs to be fixed in GPU implementation
	force -= differenceNormal * (float) ((double) G *
										 (((double) a.getMass() * (double) b.getMass()) /
										  ((double) (distance * distance))));
}

// Class that represents a Barnes-Hut tree
class BarnesHutTree {
private:
	size_t size;
	Body body;
	Cube cube;
	BarnesHutTree* octree[8] = {};
public:
	BarnesHutTree() : size(0) {}
	BarnesHutTree(Cube c) : size(0), cube(c) {}
	size_t getSize() { return size; }
	Body getBody() { return body; }
	Cube* getCube() { return &cube; }

	void insertBody(Body b) {
		body = b;
		size++;
	}

	void addBody(Body b) {
		body = add(body, b);
		size++;
	}

	void divideTree() {
		octree[0] = new BarnesHutTree(cube.getOctant(0));
		octree[1] = new BarnesHutTree(cube.getOctant(1));
		octree[2] = new BarnesHutTree(cube.getOctant(2));
		octree[3] = new BarnesHutTree(cube.getOctant(3));
		octree[4] = new BarnesHutTree(cube.getOctant(4));
		octree[5] = new BarnesHutTree(cube.getOctant(5));
		octree[6] = new BarnesHutTree(cube.getOctant(6));
		octree[7] = new BarnesHutTree(cube.getOctant(7));
	}

	BarnesHutTree* getOctree(int index) {
		return octree[index];
	}

	BarnesHutTree* octantLocation(Body b) {
		for (int i = 0; i <= 7; i++) {
			if (octree[i]->cube.within(b.getPosition())) return octree[i];
		}
	}

	glm::vec3 calculateForce(Body b) {
		glm::vec3 force(0, 0, 0);
		if (size == 0) return force; // Empty body
        if (size == 1) return gravitationalForce(body, b); // Single body
		for (int i = 0; i <= 7; i++) {
			if (octree[i]->cube.within(b.getPosition())) if(octree[i]->size != 1) force += octree[i]->calculateForce(b);
			else force += gravitationalForce(octree[i]->body, b);
		}
		return force;
	}
};

// Insert a body into a tree
void insert(BarnesHutTree* tree, Body body) {
    // Tree has already been divided
	while (tree->getSize() > 1) {
		tree->addBody(body);
		tree = tree->octantLocation(body);
	}
    // Divide tree further
	while (tree->getSize() > 0) {
		tree->addBody(body);
		tree->divideTree();
        tree = tree->octantLocation(body);
	}
    // Final insertion to unique tree
	tree->insertBody(body);
}

#endif