/**
 * AI ARENA — Game Engine v1.0.0
 * Чистый игровой движок. Никакого AI внутри.
 * 
 * Использует Cortex2Brain через коннектор.
 */

import { Cortex2Brain } from './core/cortex.js';

// ============================================================================
// UTILS
// ============================================================================

const Utils = {
    clamp(v, min, max) {
        return Math.max(min, Math.min(max, v));
    },
    
    lerp(a, b, t) {
        return a + (b - a) * this.clamp(t, 0, 1);
    },
    
    dist(a, b) {
        return Math.hypot((a?.x ?? 0) - (b?.x ?? 0), (a?.y ?? 0) - (b?.y ?? 0));
    },
    
    normalizeAngle(angle) {
        while (angle > Math.PI) angle -= Math.PI * 2;
        while (angle < -Math.PI) angle += Math.PI * 2;
        return angle;
    },
    
    angleToTarget(fromX, fromY, toX, toY, facing) {
        return this.normalizeAngle(Math.atan2(toY - fromY, toX - fromX) - facing);
    },
    
    rand(min, max) {
        return Math.random() * (max - min) + min;
    }
};

// ============================================================================
// WORLD
// ============================================================================

class World {
    constructor(width, height, seed) {
        this.width = width;
        this.height = height;
        this.seed = seed || `WORLD_${Date.now()}`;        this.walls = [];
        this._agents = new Set();
    }
    
    async init() {
        this._generateWalls(12);
        return this;
    }
    
    update(dt) {
        // Мир статичен
    }
    
    reset() {
        this.walls = [];
        this._generateWalls(12);
        this._agents.clear();
    }
    
    addAgent(agent) {
        this._agents.add(agent);
    }
    
    removeAgent(agent) {
        this._agents.delete(agent);
    }
    
    getRandomSpawnPoint(margin = 40) {
        return {
            x: Utils.rand(margin, this.width - margin),
            y: Utils.rand(margin, this.height - margin)
        };
    }
    
    _generateWalls(count) {
        const minDist = 80;
        for (let i = 0; i < count; i++) {
            let x, y, valid = false;
            for (let attempt = 0; attempt < 50 && !valid; attempt++) {
                x = Utils.rand(60, this.width - 60);
                y = Utils.rand(60, this.height - 60);
                valid = this.walls.every(w => Utils.dist({x, y}, w) > minDist);
            }
            if (valid) this.walls.push({ x, y, radius: 14 });
        }
    }
    
    render(ctx, debug = false) {
        ctx.fillStyle = '#4a4a7a';
        for (const w of this.walls) {            ctx.beginPath();
            ctx.arc(w.x, w.y, w.radius, 0, Math.PI * 2);
            ctx.fill();
            if (debug) {
                ctx.strokeStyle = '#6a6a9a';
                ctx.lineWidth = 2;
                ctx.stroke();
            }
        }
    }
}

// ============================================================================
// AGENT
// ============================================================================

class Agent {
    constructor(config) {
        this.id = config.id;
        this.x = config.x;
        this.y = config.y;
        this.color = config.color || '#ffffff';
        this.seed = config.seed;
        this.radius = config.radius || 22;
        
        this.vx = 0;
        this.vy = 0;
        this.facingAngle = 0;
        
        this.stats = {
            maxHealth: config.stats?.maxHealth || 100,
            damage: config.stats?.damage || 3,
            speed: config.stats?.speed || 22
        };
        this.health = this.stats.maxHealth;
        this.alive = true;
        
        this._throttle = 0;
        this._steering = 0;
        this._attackRequested = false;
        this._attackCooldown = 0;
        this.trail = [];
    }
    
    update(dt, world) {
        if (!this.alive) return;
        
        this._applyControls(dt);
        
        this.x += this.vx * dt * 60;        this.y += this.vy * dt * 60;
        
        this.x = Utils.clamp(this.x, this.radius, world.width - this.radius);
        this.y = Utils.clamp(this.y, this.radius, world.height - this.radius);
        
        this.vx *= 0.95;
        this.vy *= 0.95;
        
        // Trail
        if (this.trail.length === 0 || this.trail[this.trail.length - 1].age > 5) {
            this.trail.push({ x: this.x, y: this.y, age: 0 });
            if (this.trail.length > 30) this.trail.shift();
        }
        this.trail.forEach(t => t.age++);
        
        // Attack
        if (this._attackRequested && this._attackCooldown <= 0) {
            this._performAttack(world);
            this._attackCooldown = 15;
        }
        if (this._attackCooldown > 0) this._attackCooldown--;
        this._attackRequested = false;
    }
    
    _applyControls(dt) {
        const targetSpeed = this._throttle * this.stats.speed;
        const forwardX = Math.cos(this.facingAngle);
        const forwardY = Math.sin(this.facingAngle);
        
        this.vx += forwardX * (targetSpeed - this.vx) * 0.1;
        this.vy += forwardY * (targetSpeed - this.vy) * 0.1;
        
        this.facingAngle += this._steering * 0.15;
        this.facingAngle = Utils.normalizeAngle(this.facingAngle);
    }
    
    _performAttack(world) {
        const attackRange = 50;
        for (const other of world._agents) {
            if (other === this || !other.alive) continue;
            if (Utils.dist(this, other) < attackRange) {
                other.takeDamage(this.stats.damage);
            }
        }
    }
    
    setThrottle(value) {
        this._throttle = Utils.clamp(value, -1, 1);
    }
        setSteering(value) {
        this._steering = Utils.clamp(value, -1, 1);
    }
    
    requestAttack() {
        this._attackRequested = true;
    }
    
    takeDamage(amount) {
        this.health = Math.max(0, this.health - amount);
        if (this.health <= 0) this.alive = false;
    }
    
    reset(x, y) {
        this.x = x;
        this.y = y;
        this.vx = 0;
        this.vy = 0;
        this.facingAngle = 0;
        this.health = this.stats.maxHealth;
        this.alive = true;
        this.trail = [];
    }
    
    render(ctx, options = {}) {
        if (!this.alive) return;
        
        ctx.save();
        ctx.translate(this.x, this.y);
        ctx.rotate(this.facingAngle);
        
        // Body
        ctx.fillStyle = this.color;
        ctx.beginPath();
        ctx.moveTo(20, 0);
        ctx.lineTo(-15, -12);
        ctx.lineTo(-15, 12);
        ctx.closePath();
        ctx.fill();
        
        // Outline
        ctx.strokeStyle = this.health < 30 ? '#ff4444' : '#ffffff';
        ctx.lineWidth = 2;
        ctx.stroke();
        
        // HP bar
        if (this.health < this.stats.maxHealth) {
            ctx.fillStyle = '#222';
            ctx.fillRect(-20, -30, 40, 4);
            ctx.fillStyle = this.health > 50 ? '#00ff9d' : '#ff5577';            ctx.fillRect(-20, -30, 40 * (this.health / this.stats.maxHealth), 4);
        }
        
        ctx.restore();
        
        // Trail
        if (options.showTrail && this.trail.length > 1) {
            ctx.beginPath();
            ctx.moveTo(this.trail[0].x, this.trail[0].y);
            for (let i = 1; i < this.trail.length; i++) {
                ctx.globalAlpha = 1 - this.trail[i].age / 30;
                ctx.lineTo(this.trail[i].x, this.trail[i].y);
            }
            ctx.strokeStyle = this.color;
            ctx.stroke();
            ctx.globalAlpha = 1;
        }
        
        // Debug
        if (options.debug) {
            ctx.strokeStyle = 'rgba(255,255,0,0.5)';
            ctx.beginPath();
            ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
            ctx.stroke();
        }
    }
}

// ============================================================================
// GAME ENGINE
// ============================================================================

class GameEngine {
    constructor(config = {}) {
        this.config = {
            width: config.width || 820,
            height: config.height || 600,
            fps: config.fps || 60,
            seed: config.seed || `GAME_${Date.now()}`,
            ...config
        };
        
        this.world = null;
        this.agents = new Map();
        this.running = false;
        this.stepCount = 0;
        this.lastTime = 0;
        this._rafId = null;
    }
        async init() {
        this.world = new World(this.config.width, this.config.height, this.config.seed);
        await this.world.init();
        return this;
    }
    
    start() {
        if (this.running) return;
        this.running = true;
        this.lastTime = performance.now();
        this._loop();
    }
    
    stop() {
        this.running = false;
        if (this._rafId) cancelAnimationFrame(this._rafId);
    }
    
    _loop() {
        if (!this.running) return;
        const now = performance.now();
        const dt = Math.min((now - this.lastTime) / 1000, 0.1);
        this.lastTime = now;
        this.step(dt);
        this._rafId = requestAnimationFrame(() => this._loop());
    }
    
    step(dt) {
        this.stepCount++;
        this.world.update(dt);
        
        for (const agent of this.agents.values()) {
            if (agent.alive) agent.update(dt, this.world);
        }
        
        this._handleCollisions();
    }
    
    // === AGENTS ===
    
    addAgent(config) {
        const agent = new Agent({
            id: config.id || `agent_${this.agents.size}`,
            x: config.x || 400,
            y: config.y || 300,
            color: config.color || '#ffffff',
            seed: config.seed || this.config.seed,
            stats: config.stats || { maxHealth: 100, damage: 3, speed: 22 },
            ...config
        });        this.agents.set(agent.id, agent);
        this.world.addAgent(agent);
        return agent;
    }
    
    removeAgent(agentId) {
        const agent = this.agents.get(agentId);
        if (!agent) return false;
        this.world.removeAgent(agent);
        this.agents.delete(agentId);
        return true;
    }
    
    // === COLLISIONS ===
    
    _handleCollisions() {
        // Agent-agent
        for (const [idA, agentA] of this.agents) {
            if (!agentA.alive) continue;
            for (const [idB, agentB] of this.agents) {
                if (idA >= idB || !agentB.alive) continue;
                if (Utils.dist(agentA, agentB) < 44) {
                    this._resolveAgentCollision(agentA, agentB);
                }
            }
        }
        
        // Agent-wall
        for (const agent of this.agents.values()) {
            if (!agent.alive) continue;
            for (const wall of this.world.walls) {
                if (Utils.dist(agent, wall) < 36) {
                    this._resolveWallCollision(agent, wall);
                }
            }
        }
    }
    
    _resolveAgentCollision(a, b) {
        const dx = b.x - a.x, dy = b.y - a.y;
        const dist = Math.hypot(dx, dy) || 1;
        const overlap = 44 - dist;
        const pushX = (dx / dist) * overlap * 0.5;
        const pushY = (dy / dist) * overlap * 0.5;
        a.x -= pushX; a.y -= pushY;
        b.x += pushX; b.y += pushY;
        
        const repulse = 2.5;
        a.vx -= (dx / dist) * repulse;
        a.vy -= (dy / dist) * repulse;        b.vx += (dx / dist) * repulse;
        b.vy += (dy / dist) * repulse;
    }
    
    _resolveWallCollision(agent, wall) {
        const dx = agent.x - wall.x, dy = agent.y - wall.y;
        const dist = Math.hypot(dx, dy) || 1;
        const overlap = 36 - dist;
        agent.x += (dx / dist) * overlap;
        agent.y += (dy / dist) * overlap;
        
        const dot = agent.vx * dx + agent.vy * dy;
        agent.vx -= 1.5 * dot * dx / (dist * dist);
        agent.vy -= 1.5 * dot * dy / (dist * dist);
    }
    
    // === RENDER ===
    
    render(ctx, options = {}) {
        if (!ctx) return;
        
        // Background
        ctx.fillStyle = '#0a0a15';
        ctx.fillRect(0, 0, this.config.width, this.config.height);
        
        // Grid (optional)
        if (options.showGrid) {
            ctx.strokeStyle = 'rgba(50,50,90,0.2)';
            ctx.lineWidth = 1;
            for (let x = 0; x < this.config.width; x += 50) {
                ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, this.config.height); ctx.stroke();
            }
            for (let y = 0; y < this.config.height; y += 50) {
                ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(this.config.width, y); ctx.stroke();
            }
        }
        
        // World
        this.world.render(ctx, options.debug);
        
        // Agents
        for (const agent of this.agents.values()) {
            agent.render(ctx, { 
                debug: options.debug,
                showTrail: options.showTrail !== false
            });
        }
        
        // Debug info
        if (options.debug) {            ctx.fillStyle = 'rgba(100,100,140,0.8)';
            ctx.font = '10px monospace';
            ctx.fillText(`Step: ${this.stepCount} | Agents: ${this.agents.size}`, 10, 20);
        }
    }
    
    // === RESET ===
    
    reset() {
        this.stepCount = 0;
        this.world.reset();
        for (const agent of this.agents.values()) {
            const spawn = this.world.getRandomSpawnPoint();
            agent.reset(spawn.x, spawn.y);
        }
    }
}

// ============================================================================
// EXPORT
// ============================================================================

export { GameEngine, World, Agent, Utils };

if (typeof window !== 'undefined') {
    window.GameEngine = GameEngine;
    window.World = World;
    window.Agent = Agent;
    window.Utils = Utils;
          }
 
