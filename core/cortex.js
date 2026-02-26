/**
 * CORTEX2BRAIN v2.1 — Production Neural Architecture
 * 
 * FEATURES:
 * • Dynamic ratio calculation (no hardcoded constants)
 * • Input/Output schema with validation
 * • External configuration for personality/emotions
 * • Episode reset with memory preservation
 * • Debug tracing and logging
 * • Cross-platform compatibility (Browser/Node/Worker)
 * • Optional serialization with filters
 * • Batch processing for performance
 * • Clear interface definition
 * • Full JSDoc documentation
 * 
 * USAGE:
 *   import { Cortex2Brain } from './core/cortex.js';
 *   const brain = new Cortex2Brain({ seed: 'agent_1', taskSpec: {...} });
 *   brain.configurePersonality({ traits: { bravery: 0.8 } });
 *   const action = brain.forward(sensors, reward, context);
 *   brain.learn({ observation, action, reward, nextObservation, done });
 */

// ============================================================================
// TYPE DEFINITIONS (JSDoc for IDE support)
// ============================================================================

/**
 * @typedef {Object} CortexConfig
 * @property {string} [seed] - Seed for deterministic initialization
 * @property {number} [lr] - Learning rate (default: 0.015)
 * @property {number} [gamma] - TD discount factor (default: 0.99)
 * @property {number} [lambda] - Eligibility trace decay (default: 0.95)
 * @property {number} [hebbianRate] - Hebbian learning rate (default: 0.002)
 * @property {Object} [taskSpec] - Task specification for dynamic ratios
 * @property {number} [taskSpec.inputComplexity] - 0..1, how complex are inputs
 * @property {number} [taskSpec.temporalDepth] - 0..1, how much history matters
 * @property {number} [taskSpec.socialComplexity] - 0..1, how many agents interact
 */

/**
 * @typedef {Object} SensorSchema
 * @property {string} name - Sensor name
 * @property {[number, number]} range - Valid value range [min, max]
 * @property {string} [description] - Human-readable description
 * @property {'continuous'|'discrete'|'categorical'} [type] - Value type
 */

/**
 * @typedef {Object} ActionSchema * @property {string} name - Action name
 * @property {[number, number]} range - Valid value range [min, max]
 * @property {string} [description] - Human-readable description
 * @property {'continuous'|'discrete'} [type] - Value type
 */

/**
 * @typedef {Object} Experience
 * @property {number[]|Object} observation - Input sensors or named object
 * @property {number[]|Object} action - Output actions or named object
 * @property {number} reward - Scalar reward signal
 * @property {number[]|Object} [nextObservation] - Next state (for TD)
 * @property {boolean} [done] - Episode termination flag
 * @property {Object} [context] - Additional context (social, episodic, etc.)
 */

/**
 * @typedef {Object} DebugInfo
 * @property {Object} emotions - Current emotion values
 * @property {Object} personality - Personality traits and values
 * @property {Object|null} emotionalState - Active emotional commitment
 * @property {Object} stats - Learning statistics
 * @property {number[]} attentionWeights - Multi-head attention weights
 * @property {number} predictionError - Average prediction error
 * @property {Object} memoryUsage - Working/permanent/predictive memory activity
 * @property {number} cortexEnergy - Average activation in cortex layer
 */

// ============================================================================
// MAIN CLASS
// ============================================================================

export class Cortex2Brain {
    /**
     * Create a new Cortex2Brain instance
     * @param {CortexConfig} config - Configuration object
     */
    constructor(config = {}) {
        // === DIMENSIONS (fixed architecture) ===
        this.DIM = Object.freeze({
            DIM_P: 32,      // Perception layer
            DIM_A: 64,      // Attention layer (4 heads × 16)
            DIM_M: 144,     // Memory layer (3 blocks × 48)
            DIM_C: 256,     // Cortex integration layer
            DIM_S: 32,      // State/emotion layer
            DIM_D: 16,      // Decision/output layer
            DIM_SENSOR: 64, // Input sensor dimension
            NUM_HEADS: 4,   // Multi-head attention
            HEAD_DIM: 16,   // Per-head dimension
            // Memory block boundaries            M_WORK_START: 0, M_WORK_END: 48,    // Working memory
            M_PERM_START: 48, M_PERM_END: 96,   // Permanent skills
            M_PRED_START: 96, M_PRED_END: 144   // Predictive memory
        });
        
        // === CONFIGURATION ===
        this.config = {
            seed: config.seed || `CORTEX_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
            lr: config.lr ?? 0.015,
            gamma: config.gamma ?? 0.99,
            lambda: config.lambda ?? 0.95,
            hebbianRate: config.hebbianRate ?? 0.002,
            taskSpec: config.taskSpec || null,
            ...config
        };
        
        // === DYNAMIC RATIOS (calculated from taskSpec) ===
        this.RATIOS = this._calculateRatios(this.config.taskSpec);
        
        // === RANDOM NUMBER GENERATOR (seeded for reproducibility) ===
        this._rng = new SeededRNG(this.config.seed);
        
        // === WEIGHTS INITIALIZATION ===
        this._initializeWeights();
        
        // === STATE VECTORS ===
        this._initializeStateVectors();
        
        // === ELIGIBILITY TRACES (for TD learning) ===
        this._initEligibilityTraces();
        
        // === PREDICTION ERRORS ===
        this.predictionErrors = new Float32Array(48);
        
        // === STATISTICS ===
        this._stats = {
            tdErrorHistory: [],
            predictionErrorHistory: [],
            totalSteps: 0,
            avgReward: 0,
            kills: 0,
            deaths: 0,
            reflexTriggers: 0,
            consolidationEvents: 0,
            gradientClips: 0,
            validationErrors: 0
        };
        
        // === EMOTIONS (15 types with dynamics) ===
        this.emotions = this._initializeEmotions();        this.emotionHistory = [];
        this.maxHistory = 30;
        
        this.emotionalState = {
            current: null,
            history: [],
            satisfaction: 0,
            momentum: {
                fear: 0, aggression: 0, frustration: 0,
                desperation: 0, confidence: 0, vengeance: 0
            }
        };
        
        this.emotionMemory = {
            trauma: 0,
            grudge: new Map(),
            lastEmotion: null,
            unresolved: 0,
            killer: null
        };
        
        this.emotionConfig = {
            fear: { baseDuration: 180, intensityMultiplier: 2.0, satisfactionDecay: 0.02 },
            aggression: { baseDuration: 120, intensityMultiplier: 1.5, satisfactionDecay: 0.03 },
            frustration: { baseDuration: 240, intensityMultiplier: 2.5, satisfactionDecay: 0.01 },
            desperation: { baseDuration: 300, intensityMultiplier: 3.0, satisfactionDecay: 0.005 },
            confidence: { baseDuration: 150, intensityMultiplier: 1.2, satisfactionDecay: 0.04 },
            vengeance: { baseDuration: 360, intensityMultiplier: 3.5, satisfactionDecay: 0.003 }
        };
        
        this.emotionDecay = 0.98;
        this.emotionInfluence = 0.35;
        
        // === PERSONALITY (configurable traits and values) ===
        this.personality = {
            traits: {
                bravery: 0.5,      // 0 = coward, 1 = hero
                loyalty: 0.5,      // 0 = traitor, 1 = faithful
                empathy: 0.5,      // 0 = selfish, 1 = altruist
                curiosity: 0.5,    // 0 = conservative, 1 = explorer
                patience: 0.5,     // 0 = impulsive, 1 = patient
                aggression: 0.5    // 0 = peaceful, 1 = aggressive
            },
            values: {
                survival: 0.8,     // Importance of staying alive
                victory: 0.7,      // Importance of winning
                fairness: 0.4,     // Importance of fair play
                loyalty: 0.6       // Importance of being loyal
            }
        };        
        // === SOCIAL INTELLIGENCE ===
        this.socialMemory = {
            relationships: new Map(),  // agentId → { trust, affinity, history }
            reputation: 0,              // -1..1, how others perceive this agent
            observedBehaviors: new Map() // agentId → [behavior patterns]
        };
        
        // === EPISODIC MEMORY ===
        this.episodicMemory = {
            events: [],              // [{ type, agentId, emotion, reward, step, importance }]
            maxEvents: 50,           // Max events to store
            recallTrigger: 0.7,      // Threshold for memory recall
            decayRate: 0.99          // Importance decay per step
        };
        
        // === SCHEMAS (for external integration) ===
        this._inputSchema = this._generateDefaultInputSchema();
        this._outputSchema = this._generateDefaultOutputSchema();
        
        // === RUNTIME STATE ===
        this.step = 0;
        this.cumulativeReward = 0;
        this.health = 100;
        this.stats = { maxHealth: 100 };
        this._lastKiller = null;
        this._lastInputs = null;
        this._lastOutputs = null;
        this._isReady = false;
        
        // === DEBUG TRACE BUFFER ===
        this._traceBuffer = null;
        
        // Mark as initialized
        this._isReady = true;
    }
    
    // ========================================================================
    // DYNAMIC RATIO CALCULATION
    // ========================================================================
    
    /**
     * Calculate processing ratios based on task specification
     * @param {Object|null} taskSpec - Task characteristics
     * @returns {Object} Calculated ratios for signal routing
     * @private
     */
    _calculateRatios(taskSpec) {
        // Default ratios for unknown tasks
        const defaults = {            perception: 0.5,      // input → P layer
            head: 0.25,           // input → attention heads
            memoryBlock: 0.75,    // input → memory blocks
            cortex: 4.0,          // input → cortex integration
            state: 0.5            // input → state/emotion layer
        };
        
        // If no task spec, return defaults
        if (!taskSpec) return defaults;
        
        // Calculate based on task characteristics
        const {
            inputComplexity = 0.5,    // How complex/numerous are inputs
            temporalDepth = 0.5,       // How much history matters
            socialComplexity = 0.5     // How many agents interact
        } = taskSpec;
        
        return {
            // More complex inputs → more perception processing
            perception: this._clamp(0.3 + inputComplexity * 0.4, 0.3, 0.9),
            
            // More temporal depth → more memory attention
            head: this._clamp(0.15 + temporalDepth * 0.2, 0.15, 0.4),
            
            // More social complexity → more memory for relationships
            memoryBlock: this._clamp(0.5 + socialComplexity * 0.5, 0.5, 1.0),
            
            // Cortex scales with overall complexity
            cortex: this._clamp(2.0 + (inputComplexity + temporalDepth + socialComplexity) * 0.7, 2.0, 6.0),
            
            // State/emotion layer scales with social complexity
            state: this._clamp(0.3 + socialComplexity * 0.4, 0.3, 0.8)
        };
    }
    
    // ========================================================================
    // SCHEMA & VALIDATION
    // ========================================================================
    
    /**
     * Get input sensor schema
     * @returns {SensorSchema[]} Array of sensor definitions
     */
    getInputSchema() {
        return [...this._inputSchema];
    }
    
    /**
     * Get output action schema
     * @returns {ActionSchema[]} Array of action definitions     */
    getOutputSchema() {
        return [...this._outputSchema];
    }
    
    /**
     * Set custom input schema
     * @param {SensorSchema[]} schema - New sensor definitions
     */
    setInputSchema(schema) {
        if (!Array.isArray(schema)) {
            throw new Error('Input schema must be an array');
        }
        this._inputSchema = schema.map(s => ({
            name: String(s.name),
            range: Array.isArray(s.range) ? s.range : [-1, 1],
            description: s.description || '',
            type: s.type || 'continuous'
        }));
        // Update DIM_SENSOR if schema length differs
        if (schema.length > 0 && schema.length !== this.DIM.DIM_SENSOR) {
            console.warn(`Schema length (${schema.length}) differs from DIM_SENSOR (${this.DIM.DIM_SENSOR}). Truncating/padding.`);
        }
    }
    
    /**
     * Set custom output schema
     * @param {ActionSchema[]} schema - New action definitions
     */
    setOutputSchema(schema) {
        if (!Array.isArray(schema)) {
            throw new Error('Output schema must be an array');
        }
        this._outputSchema = schema.map(s => ({
            name: String(s.name),
            range: Array.isArray(s.range) ? s.range : [-1, 1],
            description: s.description || '',
            type: s.type || 'continuous'
        }));
    }
    
    /**
     * Validate and normalize input array
     * @param {number[]|Object} inputs - Raw inputs (array or named object)
     * @returns {Float32Array} Normalized sensor array
     * @throws {Error} If inputs are invalid
     */
    validateAndNormalizeInputs(inputs) {
        // Convert named object to array using schema
        if (inputs && typeof inputs === 'object' && !Array.isArray(inputs)) {            const arr = [];
            for (const sensor of this._inputSchema) {
                const val = inputs[sensor.name];
                if (val === undefined) {
                    arr.push(0); // Default to 0 if missing
                } else {
                    const [min, max] = sensor.range || [-1, 1];
                    arr.push(this._clamp((val - min) / (max - min) * 2 - 1, -1, 1));
                }
            }
            inputs = arr;
        }
        
        // Basic validation
        if (!Array.isArray(inputs)) {
            this._stats.validationErrors++;
            throw new Error(`Inputs must be array or object, got ${typeof inputs}`);
        }
        
        if (inputs.length === 0) {
            this._stats.validationErrors++;
            throw new Error('Inputs array cannot be empty');
        }
        
        // Check for NaN/Infinity
        for (let i = 0; i < Math.min(inputs.length, 20); i++) {
            if (!isFinite(inputs[i])) {
                this._stats.validationErrors++;
                console.warn(`Invalid input at index ${i}: ${inputs[i]}. Replacing with 0.`);
                inputs[i] = 0;
            }
        }
        
        // Project to cortex sensor dimension
        return this._projectToCortexSensors(inputs);
    }
    
    /**
     * Validate and clamp output array
     * @param {number[]} outputs - Raw output values
     * @returns {number[]} Clamped outputs respecting schema ranges
     */
    validateAndClampOutputs(outputs) {
        if (!Array.isArray(outputs)) {
            console.warn('Outputs must be array, returning empty');
            return [];
        }
        
        const clamped = [];
        for (let i = 0; i < outputs.length && i < this._outputSchema.length; i++) {            const [min, max] = this._outputSchema[i].range || [-1, 1];
            clamped.push(this._clamp(outputs[i], min, max));
        }
        
        // Pad with defaults if output is shorter than schema
        while (clamped.length < this._outputSchema.length) {
            const [min, max] = this._outputSchema[clamped.length].range || [-1, 1];
            clamped.push((min + max) / 2);
        }
        
        return clamped;
    }
    
    // ========================================================================
    // CONFIGURATION
    // ========================================================================
    
    /**
     * Configure personality traits and values
     * @param {Object} overrides - Partial personality configuration
     */
    configurePersonality(overrides) {
        if (overrides.traits) {
            for (const [key, val] of Object.entries(overrides.traits)) {
                if (key in this.personality.traits) {
                    this.personality.traits[key] = this._clamp(val, 0, 1);
                }
            }
        }
        if (overrides.values) {
            for (const [key, val] of Object.entries(overrides.values)) {
                if (key in this.personality.values) {
                    this.personality.values[key] = this._clamp(val, 0, 1);
                }
            }
        }
    }
    
    /**
     * Configure emotion dynamics
     * @param {Object} config - Emotion configuration overrides
     */
    configureEmotions(config) {
        if (config.emotionConfig) {
            for (const [emotion, cfg] of Object.entries(config.emotionConfig)) {
                if (emotion in this.emotionConfig) {
                    this.emotionConfig[emotion] = {
                        ...this.emotionConfig[emotion],
                        ...cfg
                    };                }
            }
        }
        if (config.decayRate !== undefined) {
            this.emotionDecay = this._clamp(config.decayRate, 0.9, 0.999);
        }
        if (config.influenceWeight !== undefined) {
            this.emotionInfluence = this._clamp(config.influenceWeight, 0.1, 0.9);
        }
    }
    
    /**
     * Update task specification and recalculate ratios
     * @param {Object} taskSpec - New task characteristics
     */
    updateTaskSpec(taskSpec) {
        this.config.taskSpec = { ...this.config.taskSpec, ...taskSpec };
        this.RATIOS = this._calculateRatios(this.config.taskSpec);
        console.log(`[Cortex2Brain] Ratios updated:`, this.RATIOS);
    }
    
    /**
     * Set learning hyperparameters
     * @param {Object} params - Learning parameters
     */
    setLearningParams(params) {
        if (params.lr !== undefined) this.lr = this._clamp(params.lr, 0.001, 0.1);
        if (params.gamma !== undefined) this.gamma = this._clamp(params.gamma, 0.9, 0.999);
        if (params.lambda !== undefined) this.lambda = this._clamp(params.lambda, 0.8, 0.99);
        if (params.hebbianRate !== undefined) this.hebbianRate = this._clamp(params.hebbianRate, 0.001, 0.01);
    }
    
    // ========================================================================
    // FORWARD PASS (with tracing support)
    // ========================================================================
    
    /**
     * Process inputs and generate actions
     * @param {number[]|Object} rawInputs - Sensor inputs (array or named object)
     * @param {number} [reward=0] - Reward signal for learning
     * @param {Object} [combatEvents=null] - Combat event metadata
     * @param {Object} [socialContext=null] - Social interaction context
     * @returns {Object} Decision result with outputs and metadata
     */
    forward(rawInputs, reward = 0, combatEvents = null, socialContext = null) {
        if (!this._isReady) {
            throw new Error('Cortex2Brain not initialized');
        }
        
        // Start trace if enabled        const trace = this._traceBuffer ? { start: performance.now(), layers: {} } : null;
        
        // Validate and normalize inputs
        const x_t = this.validateAndNormalizeInputs(rawInputs);
        const r_t = this._safeNumber(reward, 0);
        
        this._lastInputs = Array.from(x_t);
        this.cumulativeReward = this._lerp(this.cumulativeReward, r_t, 0.1);
        
        // Process social context if provided
        if (socialContext) {
            this._processSocialInput(socialContext);
            const memories = this._recallRelevantMemories({
                enemyId: socialContext.enemyId,
                healthRatio: this.health / this.stats.maxHealth,
                situation: socialContext.situation
            });
            for (const mem of memories) {
                if (mem.emotion === 'fear') this.emotions.fear = Math.min(1, this.emotions.fear + 0.1);
                if (mem.emotion === 'pride') this.emotions.confidence = Math.min(1, this.emotions.confidence + 0.1);
                if (mem.emotion === 'empathy') this.emotions.empathy = Math.min(1, this.emotions.empathy + 0.1);
            }
        }
        
        // === BUILD INPUTS FOR EACH LAYER (using dynamic RATIOS) ===
        const P_input = this._buildPInput(x_t, this.C_prev, this.S_prev, this._getMPerm(this.M_prev), this.RATIOS.perception);
        const A_inputs = [];
        for (let h = 0; h < this.DIM.NUM_HEADS; h++) {
            A_inputs.push(this._buildAInput(this.P_prev, this._getMWork(this.M_prev), this.C_prev, this.D_prev, h, this.RATIOS.head));
        }
        const M_work_input = this._buildMWorkInput(this.P_prev, this.A_prev, this.C_prev, this.RATIOS.memoryBlock);
        const M_perm_input = this._buildMPermInput(this.C_prev, this._getMPerm(this.M_prev), this.RATIOS.memoryBlock);
        const M_pred_input = this._buildMPredInput(this.A_prev, this.C_prev, this._getMWork(this.M_prev), this.D_prev, this.RATIOS.memoryBlock);
        const C_input = this._buildCInput(this.P_prev, this.A_prev, this.M_prev, this.S_prev, this.D_prev, r_t, this.RATIOS.cortex);
        const S_input = this._buildSInput(this.P_prev, this._getMPerm(this.M_prev), this.C_prev, this.D_prev, this.RATIOS.state);
        const D_input = this._buildDInput(this.C_prev, this.S_prev, this._getMPred(this.M_prev));
        
        if (trace) trace.layers.inputBuild = performance.now();
        
        // === FORWARD PROPAGATION ===
        const P_new = this._matrixMultiply(P_input, this.W_P, 'relu');
        for (let i = 0; i < this.DIM.DIM_P; i++) P_new[i] = this._leakyRelu(P_new[i] + this.b_P[i]);
        
        const A_new = new Float32Array(this.DIM.DIM_A);
        const gateProbs = this._softmax(this.W_gate, 1.0);
        for (let h = 0; h < this.DIM.NUM_HEADS; h++) {
            const headOut = this._matrixMultiply(A_inputs[h], this.W_A[h], 'relu');
            const gateWeight = gateProbs[h];
            for (let i = 0; i < this.DIM.HEAD_DIM; i++) {
                A_new[h * this.DIM.HEAD_DIM + i] = this._leakyRelu(headOut[i] + this.b_A[h][i]) * gateWeight;            }
        }
        
        const M_work_new = this._matrixMultiply(M_work_input, this.W_M_work, 'relu');
        for (let i = 0; i < 48; i++) M_work_new[i] = this._leakyRelu(M_work_new[i] + this.b_M_work[i]);
        const M_perm_new = this._matrixMultiply(M_perm_input, this.W_M_perm, 'relu');
        for (let i = 0; i < 48; i++) M_perm_new[i] = this._leakyRelu(M_perm_new[i] + this.b_M_perm[i]);
        const M_pred_new = this._matrixMultiply(M_pred_input, this.W_M_pred, 'relu');
        for (let i = 0; i < 48; i++) M_pred_new[i] = this._leakyRelu(M_pred_new[i] + this.b_M_pred[i]);
        
        const M_new = new Float32Array(this.DIM.DIM_M);
        for (let i = 0; i < 48; i++) M_new[i] = M_work_new[i];
        for (let i = 0; i < 48; i++) M_new[48 + i] = M_perm_new[i];
        for (let i = 0; i < 48; i++) M_new[96 + i] = M_pred_new[i];
        
        const C_new = this._matrixMultiply(C_input, this.W_C, 'relu');
        for (let i = 0; i < this.DIM.DIM_C; i++) C_new[i] = this._leakyRelu(C_new[i] + this.b_C[i]);
        
        const S_new = this._matrixMultiply(S_input, this.W_S, 'relu');
        for (let i = 0; i < this.DIM.DIM_S; i++) S_new[i] = this._leakyRelu(S_new[i] + this.b_S[i]);
        
        const D_new = this._matrixMultiply(D_input, this.W_D, 'tanh');
        for (let i = 0; i < this.DIM.DIM_D; i++) D_new[i] = this._tanh(D_new[i] + this.b_D[i]);
        
        if (trace) trace.layers.forward = performance.now();
        
        // === SAVE PREVIOUS STATE ===
        this.P_prev.set(this.P); this.A_prev.set(this.A); this.M_prev.set(this.M);
        this.C_prev.set(this.C); this.S_prev.set(this.S); this.D_prev.set(this.D);
        this.P.set(P_new); this.A.set(A_new); this.M.set(M_new);
        this.C.set(C_new); this.S.set(S_new); this.D.set(D_new);
        
        // === PREDICTION UPDATE ===
        this._updatePrediction(M_pred_new, x_t);
        
        // === EMOTION UPDATE ===
        this._updateEmotions(r_t, x_t, combatEvents);
        
        // === REFLEX PATH (if combat events) ===
        if (combatEvents) this._checkReflexPath(x_t, S_new, D_new);
        this._resolveCommandConflicts(D_new, this.emotions);
        
        if (trace) trace.layers.emotions = performance.now();
        
        // === PREPARE OUTPUT ===
        const output = this.validateAndClampOutputs(Array.from(this.D));
        this._lastOutputs = output;
        
        // === FINALIZE TRACE ===
        if (trace) {            trace.end = performance.now();
            trace.duration = trace.end - trace.start;
            this._traceBuffer.push(trace);
        }
        
        return {
            output,
            cortex: Array.from(this.C),
            predictionError: this._computeAvgPredError(),
            reflexTriggered: this.S.some(v => v > 0.9),
            emotions: { ...this.emotions },
            emotionalState: this.emotionalState.current ? { ...this.emotionalState.current } : null,
            personality: { ...this.personality },
            socialMemory: {
                relationships: Array.from(this.socialMemory.relationships.entries()).map(([k, v]) => [k, { 
                    trust: v.trust, 
                    affinity: v.affinity,
                    lastInteraction: v.lastInteraction
                }]),
                reputation: this.socialMemory.reputation
            },
            stats: { ...this._stats },
            trace: trace ? { ...trace } : null
        };
    }
    
    // ========================================================================
    // BATCH PROCESSING
    // ========================================================================
    
    /**
     * Process multiple inputs in batch (for parallel agents)
     * @param {Array<number[]|Object>} inputsArray - Array of input sets
     * @param {Object} [options] - Batch options
     * @param {number[]} [options.rewards] - Optional reward array
     * @returns {Object[]} Array of decision results
     */
    forwardBatch(inputsArray, options = {}) {
        const { rewards = [] } = options;
        return inputsArray.map((inputs, i) => 
            this.forward(inputs, rewards[i] || 0, null, null)
        );
    }
    
    /**
     * Learn from multiple experiences with gradient accumulation
     * @param {Experience[]} experiences - Array of experience objects
     * @param {Object} [options] - Learning options
     * @param {boolean} [options.accumulateGradients=true] - Whether to accumulate before applying
     */    learnBatch(experiences, options = {}) {
        const { accumulateGradients = true } = options;
        
        if (!accumulateGradients) {
            // Simple sequential learning
            for (const exp of experiences) {
                this.learn(exp);
            }
            return;
        }
        
        // Gradient accumulation (more stable for batches)
        const accumulated = {
            tdErrors: [],
            gradients: { P: [], C: [], D: [] }
        };
        
        for (const exp of experiences) {
            const { observation, action, reward, nextObservation, done } = exp;
            const currentValue = this._estimateValue();
            const nextValue = done ? 0 : (nextObservation ? this._estimateValue() : currentValue);
            const tdErr = this._tdError(reward, currentValue, nextValue, this.gamma);
            
            accumulated.tdErrors.push(tdErr);
            
            // Accumulate gradients (simplified - real implementation would store per-weight gradients)
            this._updateValueHead(tdErr / experiences.length);
        }
        
        // Apply accumulated updates
        const avgTDErr = accumulated.tdErrors.reduce((a, b) => a + b, 0) / experiences.length;
        this._stats.tdErrorHistory.push(Math.abs(avgTDErr));
        if (this._stats.tdErrorHistory.length > 100) this._stats.tdErrorHistory.shift();
    }
    
    // ========================================================================
    // LEARNING (single experience)
    // ========================================================================
    
    /**
     * Learn from a single experience
     * @param {Experience} experience - Experience object
     * @returns {Object} Learning metrics
     */
    learn(experience) {
        const { observation, action, reward, nextObservation, done } = experience;
        
        // Validate inputs
        const inputs = this.validateAndNormalizeInputs(observation);
        const outputs = Array.isArray(action) ? action : (action.output || []);        
        const currentValue = this._estimateValue();
        const nextValue = done ? 0 : (nextObservation ? this._estimateValue() : currentValue);
        
        return this.tdLearn(inputs, outputs, reward, currentValue, nextValue);
    }
    
    /**
     * TD-learning implementation
     * @private
     */
    tdLearn(inputs, outputs, reward, value, nextValue, actionLogProbs = null) {
        const currentValue = value !== undefined && value !== null ? this._safeNumber(value, 0) : this._estimateValue();
        const nextVal = nextValue !== undefined && nextValue !== null ? this._safeNumber(nextValue, 0) : this._estimateValue();
        const tdErr = this._tdError(reward, currentValue, nextVal, this.gamma);
        
        this._stats.tdErrorHistory.push(Math.abs(tdErr));
        if (this._stats.tdErrorHistory.length > 100) this._stats.tdErrorHistory.shift();
        
        this._updateValueHead(tdErr);
        this._updateEligibilityTraces(inputs, outputs, tdErr);
        this._applyTDError(tdErr, 'td');
        
        // Policy gradient if action probabilities provided
        if (actionLogProbs?.length) {
            const avgAdvantage = -actionLogProbs.reduce((a,b) => a+b, 0) / actionLogProbs.length;
            for (let h = 0; h < this.DIM.NUM_HEADS; h++) {
                this.W_gate[h] += this.config.lr * 0.3 * avgAdvantage * 0.25;
                this.W_gate[h] = this._clamp(this.W_gate[h], -2, 2);
            }
        }
        
        // Consolidation and Hebbian learning for significant rewards
        if (reward > 3) this._consolidateMemories(reward);
        if (reward > 1) {
            const rewardMagnitude = this._clamp(Math.abs(reward) / 5.0, 0, 1);
            const correlation = Math.sign(reward) * (0.5 + 0.5 * rewardMagnitude);
            this._hebbianUpdate(this.P_prev, this.A_prev, correlation, 'P→A');
            this._hebbianUpdate(this.A_prev, this.C_prev, correlation, 'A→C');
            this._hebbianUpdate(this.C_prev, this.D_prev, correlation, 'C→D');
        }
        
        this._stats.totalSteps++;
        this._stats.avgReward = this._lerp(this._stats.avgReward, reward, 0.01);
        
        return {
            tdError: tdErr,
            avgTDError: this._stats.tdErrorHistory.reduce((a,b)=>a+b,0)/Math.max(1,this._stats.tdErrorHistory.length),
            avgPredictionError: this._computeAvgPredError()
        };    }
    
    // ========================================================================
    // EPISODE MANAGEMENT
    // ========================================================================
    
    /**
     * Reset for new episode (preserves long-term memory, resets short-term)
     */
    resetEpisode() {
        // Reset emotional state but keep personality
        this.emotions = this._initializeEmotions();
        this.emotionalState.current = null;
        this.emotionalState.history = [];
        this.emotionHistory = [];
        
        // Reset state vectors but keep weights
        this.P.fill(0); this.A.fill(0); this.M.fill(0);
        this.C.fill(0); this.S.fill(0); this.D.fill(0);
        this.P_prev.fill(0); this.A_prev.fill(0); this.M_prev.fill(0);
        this.C_prev.fill(0); this.S_prev.fill(0); this.D_prev.fill(0);
        
        // Reset prediction errors
        this.predictionErrors.fill(0);
        
        // Reset cumulative reward
        this.cumulativeReward = 0;
        
        // Keep social memory (relationships persist between episodes)
        // But decay reputation slightly
        this.socialMemory.reputation *= 0.95;
        
        // Decay episodic memory importance
        for (const event of this.episodicMemory.events) {
            event.importance *= this.episodicMemory.decayRate;
        }
        
        // Reset step counter for episode
        this.step = 0;
        
        console.log('[Cortex2Brain] Episode reset');
    }
    
    /**
     * Full reset (including social memory)
     */
    reset() {
        this.resetEpisode();
        this.socialMemory.relationships.clear();
        this.socialMemory.reputation = 0;        this.episodicMemory.events = [];
        this._stats = {
            tdErrorHistory: [],
            predictionErrorHistory: [],
            totalSteps: 0,
            avgReward: 0,
            kills: 0,
            deaths: 0,
            reflexTriggers: 0,
            consolidationEvents: 0,
            gradientClips: 0,
            validationErrors: 0
        };
        console.log('[Cortex2Brain] Full reset');
    }
    
    // ========================================================================
    // DEBUGGING & TRACING
    // ========================================================================
    
    /**
     * Enable debug tracing
     * @param {boolean} enable - Whether to enable tracing
     * @param {number} [maxTraces=10] - Max traces to buffer
     */
    setTracing(enable, maxTraces = 10) {
        this._traceBuffer = enable ? [] : null;
        if (this._traceBuffer) {
            this._traceBuffer.maxSize = maxTraces;
        }
    }
    
    /**
     * Get comprehensive debug information
     * @returns {DebugInfo} Debug data object
     */
    getDebugInfo() {
        return {
            emotions: this.getEmotions(),
            personality: this.getPersonality(),
            emotionalState: this.emotionalState.current ? { 
                ...this.emotionalState.current,
                remainingFrames: this.emotionalState.current.remainingFrames 
            } : null,
            stats: this.getStats(),
            attentionWeights: Array.from(this.W_gate),
            predictionError: this._computeAvgPredError(),
            memoryUsage: {
                working: this._getMWork(this.M).reduce((a,b) => a + Math.abs(b), 0) / 48,
                permanent: this._getMPerm(this.M).reduce((a,b) => a + Math.abs(b), 0) / 48,                predictive: this._getMPred(this.M).reduce((a,b) => a + Math.abs(b), 0) / 48
            },
            cortexEnergy: this.C.reduce((a,b) => a + Math.abs(b), 0) / this.DIM.DIM_C,
            ratios: { ...this.RATIOS },
            trace: this._traceBuffer ? {
                count: this._traceBuffer.length,
                avgDuration: this._traceBuffer.length > 0 
                    ? this._traceBuffer.reduce((s, t) => s + t.duration, 0) / this._traceBuffer.length 
                    : 0,
                recent: this._traceBuffer.slice(-3)
            } : null
        };
    }
    
    /**
     * Get recent traces
     * @param {number} [count=5] - Number of traces to return
     * @returns {Object[]} Array of trace objects
     */
    getTraces(count = 5) {
        if (!this._traceBuffer) return [];
        return this._traceBuffer.slice(-count).map(t => ({ ...t }));
    }
    
    /**
     * Clear trace buffer
     */
    clearTraces() {
        if (this._traceBuffer) {
            this._traceBuffer = [];
        }
    }
    
    // ========================================================================
    // SERIALIZATION (with options)
    // ========================================================================
    
    /**
     * Serialize brain to JSON
     * @param {Object} [options] - Serialization options
     * @param {boolean} [options.includePersonality=true] - Include personality config
     * @param {boolean} [options.includeEmotionMemory=true] - Include emotion memory
     * @param {boolean} [options.includeSocialMemory=true] - Include social relationships
     * @param {boolean} [options.includeEpisodicMemory=false] - Include episodic events
     * @param {boolean} [options.includeStats=true] - Include learning statistics
     * @param {boolean} [options.includeSchema=true] - Include input/output schemas
     * @returns {Object} Serializable data object
     */
    toJSON(options = {}) {
        const {            includePersonality = true,
            includeEmotionMemory = true,
            includeSocialMemory = true,
            includeEpisodicMemory = false,
            includeStats = true,
            includeSchema = true
        } = options;
        
        const data = {
            type: 'Cortex2Brain',
            version: '2.1',
            config: {
                seed: this.config.seed,
                lr: this.config.lr,
                gamma: this.config.gamma,
                lambda: this.config.lambda,
                hebbianRate: this.config.hebbianRate,
                taskSpec: this.config.taskSpec,
                dims: { ...this.DIM }
            },
            weights: {
                W_P: this.W_P.map(r => Array.from(r)),
                b_P: Array.from(this.b_P),
                W_A: this.W_A.map(m => m.map(r => Array.from(r))),
                b_A: this.b_A.map(b => Array.from(b)),
                W_gate: Array.from(this.W_gate),
                W_reflex: Array.from(this.W_reflex),
                W_M_work: this.W_M_work.map(r => Array.from(r)),
                b_M_work: Array.from(this.b_M_work),
                W_M_perm: this.W_M_perm.map(r => Array.from(r)),
                b_M_perm: Array.from(this.b_M_perm),
                W_M_pred: this.W_M_pred.map(r => Array.from(r)),
                b_M_pred: Array.from(this.b_M_pred),
                W_C: this.W_C.map(r => Array.from(r)),
                b_C: Array.from(this.b_C),
                W_S: this.W_S.map(r => Array.from(r)),
                b_S: Array.from(this.b_S),
                W_D: this.W_D.map(r => Array.from(r)),
                b_D: Array.from(this.b_D),
                W_pred: this.W_pred.map(r => Array.from(r)),
                b_pred: Array.from(this.b_pred),
                W_V: Array.from(this.W_V),
                b_V: this.b_V
            }
        };
        
        if (includePersonality) {
            data.personality = { ...this.personality };
        }
                if (includeEmotionMemory) {
            data.emotionMemory = {
                trauma: this.emotionMemory.trauma,
                grudge: Array.from(this.emotionMemory.grudge.entries()),
                lastEmotion: this.emotionMemory.lastEmotion,
                unresolved: this.emotionMemory.unresolved,
                killer: this.emotionMemory.killer
            };
        }
        
        if (includeSocialMemory) {
            data.socialMemory = {
                relationships: Array.from(this.socialMemory.relationships.entries()).map(([k, v]) => [k, {
                    trust: v.trust,
                    affinity: v.affinity,
                    lastInteraction: v.lastInteraction,
                    sharedHistory: v.sharedHistory?.slice(0, 10) || []
                }]),
                reputation: this.socialMemory.reputation
            };
        }
        
        if (includeEpisodicMemory) {
            data.episodicMemory = this.episodicMemory.events.map(e => ({ ...e }));
        }
        
        if (includeStats) {
            data.stats = { ...this._stats };
        }
        
        if (includeSchema) {
            data.inputSchema = [...this._inputSchema];
            data.outputSchema = [...this._outputSchema];
        }
        
        return data;
    }
    
    /**
     * Deserialize brain from JSON
     * @param {Object} data - Serialized data object
     * @returns {Cortex2Brain} New instance with loaded state
     */
    static fromJSON(data) {
        if (!data || data.type !== 'Cortex2Brain') {
            throw new Error('Invalid Cortex2Brain JSON data');
        }
        
        const brain = new Cortex2Brain(data.config);
                // Restore weights
        const w = data.weights;
        if (w) {
            brain.W_P = w.W_P.map(r => new Float32Array(r));
            brain.b_P = new Float32Array(w.b_P);
            brain.W_A = w.W_A.map(m => m.map(r => new Float32Array(r)));
            brain.b_A = w.b_A.map(b => new Float32Array(b));
            if (w.W_gate) brain.W_gate = new Float32Array(w.W_gate);
            if (w.W_reflex) brain.W_reflex = new Float32Array(w.W_reflex);
            brain.W_M_work = w.W_M_work.map(r => new Float32Array(r));
            brain.b_M_work = new Float32Array(w.b_M_work);
            brain.W_M_perm = w.W_M_perm.map(r => new Float32Array(r));
            brain.b_M_perm = new Float32Array(w.b_M_perm);
            brain.W_M_pred = w.W_M_pred.map(r => new Float32Array(r));
            brain.b_M_pred = new Float32Array(w.b_M_pred);
            brain.W_C = w.W_C.map(r => new Float32Array(r));
            brain.b_C = new Float32Array(w.b_C);
            brain.W_S = w.W_S.map(r => new Float32Array(r));
            brain.b_S = new Float32Array(w.b_S);
            brain.W_D = w.W_D.map(r => new Float32Array(r));
            brain.b_D = new Float32Array(w.b_D);
            brain.W_pred = w.W_pred.map(r => new Float32Array(r));
            brain.b_pred = new Float32Array(w.b_pred);
            if (w.W_V) {
                brain.W_V = new Float32Array(w.W_V);
                brain.b_V = w.b_V ?? 0;
            }
        }
        
        // Restore personality
        if (data.personality) {
            brain.personality.traits = { ...brain.personality.traits, ...data.personality.traits };
            brain.personality.values = { ...brain.personality.values, ...data.personality.values };
        }
        
        // Restore emotion memory
        if (data.emotionMemory) {
            brain.emotionMemory.trauma = data.emotionMemory.trauma || 0;
            brain.emotionMemory.grudge = new Map(data.emotionMemory.grudge || []);
            brain.emotionMemory.lastEmotion = data.emotionMemory.lastEmotion;
            brain.emotionMemory.unresolved = data.emotionMemory.unresolved || 0;
            brain.emotionMemory.killer = data.emotionMemory.killer;
        }
        
        // Restore social memory
        if (data.socialMemory) {
            brain.socialMemory.reputation = data.socialMemory.reputation ?? 0;
            if (data.socialMemory.relationships) {
                for (const [id, rel] of data.socialMemory.relationships) {
                    brain.socialMemory.relationships.set(id, {                        trust: rel.trust ?? 0,
                        affinity: rel.affinity ?? 0,
                        lastInteraction: rel.lastInteraction ?? 0,
                        sharedHistory: rel.sharedHistory || []
                    });
                }
            }
        }
        
        // Restore episodic memory
        if (data.episodicMemory) {
            brain.episodicMemory.events = data.episodicMemory.map(e => ({ ...e }));
        }
        
        // Restore schemas
        if (data.inputSchema) brain._inputSchema = data.inputSchema;
        if (data.outputSchema) brain._outputSchema = data.outputSchema;
        
        // Restore stats
        if (data.stats) brain._stats = { ...brain._stats, ...data.stats };
        
        return brain;
    }
    
    // ========================================================================
    // CLONING & MUTATION (for evolution)
    // ========================================================================
    
    /**
     * Create a clone of this brain
     * @param {Object} [options] - Clone options
     * @param {boolean} [options.copyWeights=true] - Copy weight matrices
     * @param {boolean} [options.copyState=false] - Copy current state vectors
     * @param {boolean} [options.copyMemory=true] - Copy memory and relationships
     * @returns {Cortex2Brain} New cloned instance
     */
    clone(options = {}) {
        const {
            copyWeights = true,
            copyState = false,
            copyMemory = true
        } = options;
        
        const clone = new Cortex2Brain({
            seed: this.config.seed + '_clone_' + Date.now(),
            lr: this.config.lr,
            gamma: this.config.gamma,
            lambda: this.config.lambda,
            hebbianRate: this.config.hebbianRate,
            taskSpec: this.config.taskSpec ? { ...this.config.taskSpec } : null        });
        
        // Copy weights if requested
        if (copyWeights) {
            const copyMat = (src) => src.map(r => new Float32Array(r));
            clone.W_P = copyMat(this.W_P);
            clone.b_P = new Float32Array(this.b_P);
            clone.W_A = this.W_A.map(m => m.map(r => new Float32Array(r)));
            clone.b_A = this.b_A.map(b => new Float32Array(b));
            clone.W_gate = new Float32Array(this.W_gate);
            clone.W_reflex = new Float32Array(this.W_reflex);
            clone.W_M_work = copyMat(this.W_M_work);
            clone.b_M_work = new Float32Array(this.b_M_work);
            clone.W_M_perm = copyMat(this.W_M_perm);
            clone.b_M_perm = new Float32Array(this.b_M_perm);
            clone.W_M_pred = copyMat(this.W_M_pred);
            clone.b_M_pred = new Float32Array(this.b_M_pred);
            clone.W_C = copyMat(this.W_C);
            clone.b_C = new Float32Array(this.b_C);
            clone.W_S = copyMat(this.W_S);
            clone.b_S = new Float32Array(this.b_S);
            clone.W_D = copyMat(this.W_D);
            clone.b_D = new Float32Array(this.b_D);
            clone.W_pred = copyMat(this.W_pred);
            clone.b_pred = new Float32Array(this.b_pred);
            clone.W_V = new Float32Array(this.W_V);
            clone.b_V = this.b_V;
        }
        
        // Copy state vectors if requested
        if (copyState) {
            clone.P.set(this.P); clone.A.set(this.A); clone.M.set(this.M);
            clone.C.set(this.C); clone.S.set(this.S); clone.D.set(this.D);
            clone.P_prev.set(this.P_prev); clone.A_prev.set(this.A_prev); clone.M_prev.set(this.M_prev);
            clone.C_prev.set(this.C_prev); clone.S_prev.set(this.S_prev); clone.D_prev.set(this.D_prev);
            clone.predictionErrors.set(this.predictionErrors);
        }
        
        // Copy personality
        clone.personality = {
            traits: { ...this.personality.traits },
            values: { ...this.personality.values }
        };
        
        // Copy memory if requested
        if (copyMemory) {
            clone.emotions = { ...this.emotions };
            clone.emotionHistory = [...this.emotionHistory];
            clone.emotionalState = {
                current: this.emotionalState.current ? { ...this.emotionalState.current } : null,                history: [...this.emotionalState.history],
                satisfaction: this.emotionalState.satisfaction,
                momentum: { ...this.emotionalState.momentum }
            };
            clone.emotionMemory = {
                trauma: this.emotionMemory.trauma,
                grudge: new Map(this.emotionMemory.grudge),
                lastEmotion: this.emotionMemory.lastEmotion ? { ...this.emotionMemory.lastEmotion } : null,
                unresolved: this.emotionMemory.unresolved,
                killer: this.emotionMemory.killer
            };
            clone.socialMemory = {
                relationships: new Map(this.socialMemory.relationships),
                reputation: this.socialMemory.reputation,
                observedBehaviors: new Map(this.socialMemory.observedBehaviors)
            };
            clone.episodicMemory = {
                events: [...this.episodicMemory.events],
                maxEvents: this.episodicMemory.maxEvents,
                recallTrigger: this.episodicMemory.recallTrigger,
                decayRate: this.episodicMemory.decayRate
            };
        }
        
        // Copy schemas
        clone._inputSchema = [...this._inputSchema];
        clone._outputSchema = [...this._outputSchema];
        
        // Copy stats
        clone._stats = { ...this._stats, tdErrorHistory: [...this._stats.tdErrorHistory] };
        
        return clone;
    }
    
    /**
     * Mutate weights for evolutionary algorithms
     * @param {number} [rate=0.01] - Mutation probability per weight
     * @param {number} [strength=0.1] - Mutation magnitude
     * @returns {number} Number of mutations applied
     */
    mutate(rate = 0.01, strength = 0.1) {
        let mutations = 0;
        
        const mutateMat = (mat) => {
            if (!mat?.length) return;
            for (let i = 0; i < mat.length; i++) {
                if (!mat[i]?.length) continue;
                for (let j = 0; j < mat[i].length; j++) {
                    if (this._rng.next() < rate) {
                        mat[i][j] += this._rng.gaussian(0, strength);                        mat[i][j] = this._clamp(mat[i][j], -2, 2);
                        mutations++;
                    }
                }
            }
        };
        
        mutateMat(this.W_P);
        mutateMat(this.W_C);
        mutateMat(this.W_D);
        for (const W of this.W_A) mutateMat(W);
        mutateMat(this.W_M_work);
        mutateMat(this.W_M_perm);
        mutateMat(this.W_M_pred);
        mutateMat(this.W_S);
        mutateMat(this.W_pred);
        
        // Mutate gate weights
        for (let i = 0; i < this.W_gate.length; i++) {
            if (this._rng.next() < rate) {
                this.W_gate[i] += this._rng.gaussian(0, strength);
                this.W_gate[i] = this._clamp(this.W_gate[i], -2, 2);
                mutations++;
            }
        }
        
        return mutations;
    }
    
    // ========================================================================
    // GETTERS
    // ========================================================================
    
    /** @returns {Object} Current emotion values */
    getEmotions() { return { ...this.emotions }; }
    
    /** @returns {Object} Personality configuration */
    getPersonality() { return { ...this.personality }; }
    
    /** @returns {Object} Learning statistics */
    getStats() { return { ...this._stats }; }
    
    /** @returns {number} Input dimension */
    getInputDim() { return this.DIM.DIM_SENSOR; }
    
    /** @returns {number} Output dimension */
    getOutputDim() { return this.DIM.DIM_D; }
    
    /** @returns {boolean} Whether brain is ready for use */
    isReady() { return this._isReady; }    
    // ========================================================================
    // PRIVATE HELPERS
    // ========================================================================
    
    _initializeWeights() {
        this.W_P = this._initMatrix(400, this.DIM.DIM_P, 0.3);
        this.b_P = this._initVector(this.DIM.DIM_P, 0.1);
        
        this.W_A = [];
        this.b_A = [];
        for (let h = 0; h < this.DIM.NUM_HEADS; h++) {
            this.W_A.push(this._initMatrix(160, this.DIM.HEAD_DIM, 0.3));
            this.b_A.push(this._initVector(this.DIM.HEAD_DIM, 0.1));
        }
        
        this.W_gate = this._initVector(this.DIM.NUM_HEADS, 0.25);
        this.W_reflex = this._initVector(4, 0.5);
        
        this.W_M_work = this._initMatrix(160, 48, 0.3);
        this.b_M_work = this._initVector(48, 0.1);
        this.W_M_perm = this._initMatrix(304, 48, 0.3);
        this.b_M_perm = this._initVector(48, 0.1);
        this.W_M_pred = this._initMatrix(192, 48, 0.3);
        this.b_M_pred = this._initVector(48, 0.1);
        
        this.W_C = this._initMatrix(289, this.DIM.DIM_C, 0.25);
        this.b_C = this._initVector(this.DIM.DIM_C, 0.1);
        
        this.W_S = this._initMatrix(128, this.DIM.DIM_S, 0.3);
        this.b_S = this._initVector(this.DIM.DIM_S, 0.1);
        this.W_D = this._initMatrix(336, this.DIM.DIM_D, 0.2);
        this.b_D = this._initVector(this.DIM.DIM_D, 0.1);
        
        this.W_pred = this._initMatrix(48, this.DIM.DIM_SENSOR, 0.15);
        this.b_pred = this._initVector(this.DIM.DIM_SENSOR, 0.05);
        
        this.W_V = this._initVector(this.DIM.DIM_C, 0.1);
        this.b_V = 0;
    }
    
    _initializeStateVectors() {
        this.P = new Float32Array(this.DIM.DIM_P);
        this.A = new Float32Array(this.DIM.DIM_A);
        this.M = new Float32Array(this.DIM.DIM_M);
        this.C = new Float32Array(this.DIM.DIM_C);
        this.S = new Float32Array(this.DIM.DIM_S);
        this.D = new Float32Array(this.DIM.DIM_D);
        
        this.P_prev = new Float32Array(this.DIM.DIM_P);        this.A_prev = new Float32Array(this.DIM.DIM_A);
        this.M_prev = new Float32Array(this.DIM.DIM_M);
        this.C_prev = new Float32Array(this.DIM.DIM_C);
        this.S_prev = new Float32Array(this.DIM.DIM_S);
        this.D_prev = new Float32Array(this.DIM.DIM_D);
    }
    
    _initEligibilityTraces() {
        this.trace_P = new Float32Array(400 * this.DIM.DIM_P);
        this.trace_C = new Float32Array(289 * this.DIM.DIM_C);
        this.trace_D = new Float32Array(336 * this.DIM.DIM_D);
        this.trace_pred = new Float32Array(48 * this.DIM.DIM_SENSOR);
    }
    
    _initializeEmotions() {
        return {
            threat: 0, frustration: 0, confidence: 0, aggression: 0,
            fear: 0, surprise: 0, caution: 0, desperation: 0, vengeance: 0,
            empathy: 0, loyalty: 0, shame: 0, pride: 0, curiosity: 0, boredom: 0
        };
    }
    
    _generateDefaultInputSchema() {
        return [
            { name: 'enemy_x', range: [-1, 1], description: 'Enemy X position (normalized)' },
            { name: 'enemy_y', range: [-1, 1], description: 'Enemy Y position (normalized)' },
            { name: 'distance', range: [0, 1], description: 'Distance to nearest enemy' },
            { name: 'enemy_health', range: [0, 1], description: 'Enemy health ratio' },
            { name: 'wall_x', range: [-1, 1], description: 'Nearest wall X' },
            { name: 'wall_y', range: [-1, 1], description: 'Nearest wall Y' },
            { name: 'wall_distance', range: [0, 1], description: 'Distance to nearest wall' },
            { name: 'self_x', range: [-1, 1], description: 'Self X position' },
            { name: 'self_y', range: [-1, 1], description: 'Self Y position' },
            { name: 'self_health', range: [0, 1], description: 'Self health ratio' },
            { name: 'self_reward', range: [-1, 1], description: 'Cumulative reward signal' },
            { name: 'self_damage', range: [0, 1], description: 'Self damage stat' },
            { name: 'self_dodge', range: [0, 1], description: 'Self dodge chance' },
            { name: 'self_fireRate', range: [0, 1], description: 'Self fire rate' },
            { name: 'step_normalized', range: [0, 1], description: 'Normalized step count' },
            // Weapon one-hot encoding (8 weapons)
            { name: 'weapon_hammer', range: [0, 1], type: 'discrete' },
            { name: 'weapon_drill', range: [0, 1], type: 'discrete' },
            { name: 'weapon_sword', range: [0, 1], type: 'discrete' },
            { name: 'weapon_blaster', range: [0, 1], type: 'discrete' },
            { name: 'weapon_axe', range: [0, 1], type: 'discrete' },
            { name: 'weapon_dagger', range: [0, 1], type: 'discrete' },
            { name: 'weapon_spear', range: [0, 1], type: 'discrete' },
            { name: 'weapon_chainsaw', range: [0, 1], type: 'discrete' },
            { name: 'death_count', range: [0, 1], description: 'Normalized death count' },
            { name: 'forced_combat', range: [0, 1], type: 'discrete', description: 'Forced combat flag' },            // Tactical inputs (18 values)
            { name: 'angle_to_enemy', range: [-1, 1], description: 'Angle to enemy relative to facing' },
            { name: 'enemy_sees_me', range: [-1, 1], description: 'Whether enemy is facing me' },
            { name: 'in_enemy_arc', range: [0, 1], type: 'discrete', description: 'Am I in enemy weapon arc' },
            { name: 'enemy_in_my_arc', range: [0, 1], type: 'discrete', description: 'Is enemy in my weapon arc' },
            { name: 'threat_level', range: [0, 1], description: 'Assessed threat level' },
            { name: 'flank_vector_x', range: [-1, 1], description: 'Flank direction X' },
            { name: 'flank_vector_y', range: [-1, 1], description: 'Flank direction Y' },
            { name: 'angle_from_enemy', range: [-1, 1], description: 'My angle relative to enemy facing' },
            { name: 'dist_to_arc_edge', range: [-1, 1], description: 'Distance to edge of enemy arc' },
            { name: 'self_damage_ratio', range: [0, 1], description: 'Damage taken ratio' },
            { name: 'predicted_enemy_x', range: [-1, 1], description: 'Predicted enemy X' },
            { name: 'predicted_enemy_y', range: [-1, 1], description: 'Predicted enemy Y' },
            { name: 'path_blocked', range: [0, 1], type: 'discrete', description: 'Is flank path blocked' },
            { name: 'recent_exposure', range: [0, 1], description: 'Recent exposure to enemy arc' },
            { name: 'exposure_level', range: [0, 1], description: 'Current exposure level' },
            { name: 'in_combat_range', range: [0, 1], type: 'discrete', description: 'Within combat range' },
            { name: 'wall_dist_normalized', range: [0, 1], description: 'Normalized distance to nearest wall' },
            { name: 'wall_avoidance_steer', range: [-1, 1], description: 'Recommended wall avoidance steer' }
        ];
    }
    
    _generateDefaultOutputSchema() {
        return [
            { name: 'output_0', range: [-1, 1], description: 'Generic output 0' },
            { name: 'output_1', range: [-1, 1], description: 'Generic output 1' },
            { name: 'output_2', range: [-1, 1], description: 'Generic output 2' },
            { name: 'output_3', range: [-1, 1], description: 'Generic output 3' },
            { name: 'throttle', range: [-1, 1], description: 'Movement throttle (-1=reverse, 1=forward)' },
            { name: 'curiosity', range: [0, 1], description: 'Exploration drive' },
            { name: 'aggression', range: [0, 1], description: 'Aggression level' },
            { name: 'memory_weight', range: [0, 1], description: 'Weight given to memory vs perception' },
            { name: 'steering', range: [-1, 1], description: 'Steering direction' },
            { name: 'circular_bias', range: [-1, 1], description: 'Circular movement bias' },
            { name: 'retreat_urgency', range: [0, 1], description: 'Urgency to retreat' },
            { name: 'flank_commit', range: [0, 1], description: 'Commitment to flanking maneuver' },
            { name: 'output_12', range: [-1, 1], description: 'Generic output 12' },
            { name: 'output_13', range: [-1, 1], description: 'Generic output 13' },
            { name: 'output_14', range: [-1, 1], description: 'Generic output 14' },
            { name: 'output_15', range: [-1, 1], description: 'Generic output 15' }
        ];
    }
    
    _initMatrix(rows, cols, scale) {
        const mat = [];
        for (let i = 0; i < rows; i++) {
            const row = new Float32Array(cols);
            for (let j = 0; j < cols; j++) {
                const hash = this._hashSeed(this.config.seed, i * 10000 + j, 0);
                row[j] = (hash * 2 - 1) * scale;            }
            mat.push(row);
        }
        return mat;
    }
    
    _initVector(length, scale) {
        const vec = new Float32Array(length);
        for (let i = 0; i < length; i++) {
            const hash = this._hashSeed(this.config.seed, i, 1000);
            vec[i] = (hash * 2 - 1) * scale;
        }
        return vec;
    }
    
    _hashSeed(seed, i, j) {
        let h = String(seed).split('').reduce((a, c, idx) => {
            return ((a * 31 + c.charCodeAt(0)) ^ (idx * 17 + i * 7919 + j * 104729)) % 2147483647;
        }, String(seed).length) || 12345;
        h = (h * 2654435761) >>> 0;
        return h / 4294967296;
    }
    
    _clamp(v, min, max) { return Math.max(min, Math.min(max, v)); }
    _lerp(a, b, t) { return a + (b - a) * this._clamp(t, 0, 1); }
    _safeNumber(val, fallback = 0) { return (typeof val === 'number' && isFinite(val)) ? val : fallback; }
    _tanh(x) { if (x > 20) return 1; if (x < -20) return -1; const e = Math.exp(2 * x); return (e - 1) / (e + 1); }
    _leakyRelu(x, α = 0.01) { return x > 0 ? x : α * x; }
    
    _softmax(arr, temp = 1) {
        if (!arr || !Array.isArray(arr) || arr.length === 0) return [];
        const max = Math.max(...arr);
        const exps = arr.map(v => Math.exp(Math.min((v - max) / temp, 700)));
        const sum = exps.reduce((s, x) => s + x, 0) || 1e-10;
        return exps.map(e => e / sum);
    }
    
    _matrixMultiply(vec, mat, activation = 'tanh') {
        if (!vec?.length || !mat?.length || !mat[0]?.length) return new Float32Array(0);
        const out = new Float32Array(mat[0].length);
        for (let i = 0; i < mat[0].length; i++) {
            let sum = 0;
            for (let j = 0; j < vec.length; j++) {
                sum += this._safeNumber(vec[j], 0) * this._safeNumber(mat[j]?.[i], 0);
            }
            switch (activation) {
                case 'tanh': out[i] = this._tanh(sum); break;
                case 'relu': out[i] = this._leakyRelu(sum); break;
                default: out[i] = this._tanh(sum);
            }        }
        return out;
    }
    
    _projectToCortexSensors(rawInputs) {
        const output = new Float32Array(this.DIM.DIM_SENSOR);
        const src = rawInputs || [];
        const srcLen = Math.min(src.length, this.DIM.DIM_SENSOR);
        for (let i = 0; i < srcLen; i++) {
            output[i] = this._clamp(this._safeNumber(src[i], 0), -1, 1);
        }
        for (let i = srcLen; i < this.DIM.DIM_SENSOR; i++) {
            const hash = this._hashSeed(this.config.seed, i, 0xDEADBEEF);
            output[i] = (hash * 2 - 1) * 0.05;
        }
        return output;
    }
    
    _getMWork(M) { return M.slice(this.DIM.M_WORK_START, this.DIM.M_WORK_END); }
    _getMPerm(M) { return M.slice(this.DIM.M_PERM_START, this.DIM.M_PERM_END); }
    _getMPred(M) { return M.slice(this.DIM.M_PRED_START, this.DIM.M_PRED_END); }
    
    // Build methods with ratio parameter for dynamic signal routing
    _buildPInput(x_t, C_prev, S_prev, M_perm_prev, ratio = 0.5) {
        const input = new Float32Array(400);
        let idx = 0;
        const sensorWeight = ratio;
        for (let i = 0; i < this.DIM.DIM_SENSOR; i++) input[idx++] = x_t[i] * sensorWeight;
        for (let i = 0; i < this.DIM.DIM_C; i++) input[idx++] = C_prev[i];
        for (let i = 0; i < this.DIM.DIM_S; i++) input[idx++] = S_prev[i];
        for (let i = 0; i < 48; i++) input[idx++] = M_perm_prev[i];
        return input;
    }
    
    _buildAInput(P_t, M_work, C_prev, D_prev, headIdx, ratio = 0.25) {
        const input = new Float32Array(160);
        let idx = 0;
        for (let i = 0; i < this.DIM.DIM_P; i++) input[idx++] = P_t[i] * ratio;
        for (let i = 0; i < 48; i++) input[idx++] = M_work[i];
        const cStart = headIdx * 64;
        for (let i = 0; i < 64; i++) {
            const cIdx = cStart + i;
            input[idx++] = (cIdx < this.DIM.DIM_C) ? C_prev[cIdx] : 0;
        }
        for (let i = 0; i < this.DIM.DIM_D; i++) input[idx++] = D_prev[i];
        return input;
    }
    
    _buildMWorkInput(P_t, A_t, C_prev, ratio = 0.75) {
        const input = new Float32Array(160);        let idx = 0;
        for (let i = 0; i < this.DIM.DIM_P; i++) input[idx++] = P_t[i] * ratio;
        for (let i = 0; i < this.DIM.DIM_A; i++) input[idx++] = A_t[i] * ratio;
        for (let i = 0; i < 64; i++) input[idx++] = (i < this.DIM.DIM_C) ? C_prev[i] : 0;
        return input;
    }
    
    _buildMPermInput(C_prev, M_perm_prev, ratio = 0.75) {
        const input = new Float32Array(304);
        let idx = 0;
        for (let i = 0; i < this.DIM.DIM_C; i++) input[idx++] = C_prev[i] * ratio;
        for (let i = 0; i < 48; i++) input[idx++] = M_perm_prev[i];
        return input;
    }
    
    _buildMPredInput(A_t, C_prev, M_work, D_prev, ratio = 0.75) {
        const input = new Float32Array(192);
        let idx = 0;
        for (let i = 0; i < this.DIM.DIM_A; i++) input[idx++] = A_t[i] * ratio;
        for (let i = 0; i < 64; i++) {
            const cIdx = 192 + i;
            input[idx++] = (cIdx < this.DIM.DIM_C) ? C_prev[cIdx] : 0;
        }
        for (let i = 0; i < 48; i++) input[idx++] = M_work[i];
        for (let i = 0; i < this.DIM.DIM_D; i++) input[idx++] = D_prev[i];
        return input;
    }
    
    _buildCInput(P_t, A_t, M_t, S_prev, D_prev, r_t, ratio = 4.0) {
        const input = new Float32Array(289);
        let idx = 0;
        for (let i = 0; i < this.DIM.DIM_P; i++) input[idx++] = P_t[i] * ratio;
        for (let i = 0; i < this.DIM.DIM_A; i++) input[idx++] = A_t[i] * ratio;
        for (let i = 0; i < this.DIM.DIM_M; i++) input[idx++] = M_t[i] * ratio;
        for (let i = 0; i < this.DIM.DIM_S; i++) input[idx++] = S_prev[i];
        for (let i = 0; i < this.DIM.DIM_D; i++) input[idx++] = D_prev[i];
        input[idx] = r_t;
        return input;
    }
    
    _buildSInput(P_t, M_perm, C_prev, D_prev, ratio = 0.5) {
        const input = new Float32Array(128);
        let idx = 0;
        for (let i = 0; i < this.DIM.DIM_P; i++) input[idx++] = P_t[i] * ratio;
        for (let i = 0; i < 48; i++) input[idx++] = M_perm[i];
        for (let i = 0; i < 32; i++) {
            const cIdx = 224 + i;
            input[idx++] = (cIdx < this.DIM.DIM_C) ? C_prev[cIdx] : 0;
        }
        for (let i = 0; i < this.DIM.DIM_D; i++) input[idx++] = D_prev[i] * ratio;        return input;
    }
    
    _buildDInput(C_t, S_t, M_pred) {
        const input = new Float32Array(336);
        let idx = 0;
        for (let i = 0; i < this.DIM.DIM_C; i++) input[idx++] = C_t[i];
        for (let i = 0; i < this.DIM.DIM_S; i++) input[idx++] = S_t[i];
        for (let i = 0; i < 48; i++) input[idx++] = M_pred[i];
        return input;
    }
    
    _updatePrediction(M_pred, x_t) {
        const x_pred = this._linearMultiply(M_pred, this.W_pred);
        for (let i = 0; i < this.DIM.DIM_SENSOR; i++) x_pred[i] += this.b_pred[i];
        for (let i = 0; i < 48; i++) {
            const error = Math.abs(x_pred[i] - x_t[i]);
            this.predictionErrors[i] = this._lerp(this.predictionErrors[i], error, 0.1);
        }
    }
    
    _linearMultiply(vec, mat) {
        if (!vec?.length || !mat?.length || !mat[0]?.length) return new Float32Array(0);
        const out = new Float32Array(mat[0].length);
        for (let i = 0; i < mat[0].length; i++) {
            for (let j = 0; j < vec.length; j++) {
                out[i] += this._safeNumber(vec[j], 0) * this._safeNumber(mat[j]?.[i], 0);
            }
        }
        return out;
    }
    
    _computeAvgPredError() {
        let sum = 0;
        for (let i = 0; i < 48; i++) sum += this.predictionErrors[i];
        return sum / 48;
    }
    
    _estimateValue() {
        let sum = this.b_V;
        for (let i = 0; i < this.DIM.DIM_C; i++) sum += this.C[i] * this.W_V[i];
        return this._tanh(sum);
    }
    
    _updateValueHead(tdErr) {
        const alpha = this.config.lr * 0.5;
        for (let i = 0; i < this.DIM.DIM_C; i++) {
            this.W_V[i] += alpha * tdErr * this.C[i];
            this.W_V[i] = this._clamp(this.W_V[i], -2, 2);
        }        this.b_V += alpha * tdErr;
    }
    
    _updateEligibilityTraces(inputs, outputs, tdErr) {
        const decay = this.config.gamma * this.config.lambda;
        for (let i = 0; i < 400; i++) {
            for (let j = 0; j < this.DIM.DIM_P; j++) {
                const idx = i * this.DIM.DIM_P + j;
                this.trace_P[idx] = decay * this.trace_P[idx] + inputs[i] * this.P[j];
                this.trace_P[idx] = this._clamp(this.trace_P[idx], -10, 10);
            }
        }
    }
    
    _applyTDError(tdErr, errorType = 'td') {
        const alpha = this.config.lr * Math.sign(tdErr) * 0.1;
        const MAX_GRAD = 0.3;
        for (let i = 64; i < 128; i++) {
            for (let j = 0; j < 289; j++) {
                if (this.W_C[j] && this.W_C[j][i] !== undefined) {
                    const delta = this._clamp(alpha * 0.01 * this.trace_C[j * this.DIM.DIM_C + i], -MAX_GRAD, MAX_GRAD);
                    this.W_C[j][i] += delta;
                    this.W_C[j][i] = this._clamp(this.W_C[j][i], -2, 2);
                }
            }
        }
    }
    
    _hebbianUpdate(pre, post, correlation, label) {
        if (!pre?.length || !post?.length) return;
        const eta = this.config.hebbianRate * correlation * 0.5;
        const step = Math.max(1, Math.floor(pre.length / 50));
        for (let i = 0; i < post.length; i += step) {
            for (let j = 0; j < pre.length; j += step) {
                const delta = eta * pre[j] * post[i];
                if (label === 'P→A' && i < this.DIM.DIM_A) {
                    const headIdx = Math.floor(i / this.DIM.HEAD_DIM);
                    const localI = i % this.DIM.HEAD_DIM;
                    if (j < 160 && this.W_A[headIdx] && this.W_A[headIdx][j]) {
                        this.W_A[headIdx][j][localI] += delta;
                        this.W_A[headIdx][j][localI] = this._clamp(this.W_A[headIdx][j][localI], -2, 2);
                    }
                }
            }
        }
    }
    
    _consolidateMemories(reward) {
        for (let i = 0; i < 48; i++) {
            if (this.predictionErrors[i] > 0.25) {                this.episodicMemory.events.push({
                    idx: i,
                    value: this.M[96 + i],
                    importance: this.predictionErrors[i] * reward,
                    step: this.step
                });
            }
        }
        if (this.episodicMemory.events.length > this.episodicMemory.maxEvents) {
            this.episodicMemory.events.sort((a, b) => b.importance - a.importance);
            this.episodicMemory.events.length = this.episodicMemory.maxEvents;
        }
    }
    
    _tdError(reward, value, nextValue, gamma = 0.99) {
        return this._safeNumber(reward, 0) + gamma * this._safeNumber(nextValue, 0) - this._safeNumber(value, 0);
    }
    
    // ========================================================================
    // EMOTION & SOCIAL METHODS (existing implementations)
    // ========================================================================
    
    _updateEmotions(r_t, x_t, combatEvents = null) {
        this.step++;
        this.emotionHistory.push({ ...this.emotions, step: this.step });
        if (this.emotionHistory.length > this.maxHistory) this.emotionHistory.shift();
        if (combatEvents) this._processCombatEvents(combatEvents);
        
        if (this.emotionalState.current) {
            const state = this.emotionalState.current;
            const config = this.emotionConfig[state.name];
            state.remainingFrames--;
            state.satisfaction = this._lerp(state.satisfaction, this.emotionalState.satisfaction, 0.1);
            if (this._canExitEmotion(state)) this._exitEmotion(state);
            if (this._shouldExtendEmotion(state)) state.remainingFrames = Math.min(state.remainingFrames + 30, config.baseDuration * 3);
            const transition = this._checkEmotionTransition(state);
            if (transition) this._enterEmotion(transition.name, transition.intensity);
        } else {
            const dominant = this._getDominantEmotion();
            if (dominant.intensity > 0.7) this._enterEmotion(dominant.name, dominant.intensity);
        }
        
        if (r_t < 0.1) this.emotions.frustration = Math.min(1, this.emotions.frustration + 0.03);
        else this.emotions.frustration *= 0.92;
        
        const normalizedReward = this._clamp((this.cumulativeReward + 100) / 200, 0, 1);
        if (normalizedReward < 0.3) this.emotions.desperation = Math.min(1, this.emotions.desperation + 0.05);
        else this.emotions.desperation *= 0.85;
        
        if (r_t > 1) this.emotions.confidence = Math.min(1, this.emotions.confidence + 0.05);        else this.emotions.confidence *= 0.95;
        
        if (r_t < -0.5) this.emotions.fear = Math.min(1, this.emotions.fear + 0.1);
        else this.emotions.fear *= 0.85;
        
        this.emotions.surprise = this._computeAvgPredError();
        const healthInput = x_t[12] ?? 0.5;
        this.emotions.caution = 1 - healthInput;
        const distanceInput = x_t[2] ?? 0.5;
        this.emotions.threat = 1 - distanceInput;
        
        const recentFrames = Math.min(10, this.emotionHistory.length);
        if (recentFrames > 0) {
            const recent = this.emotionHistory.slice(-recentFrames);
            this.emotionalState.momentum.fear = recent.reduce((s, e) => s + e.fear, 0) / recentFrames;
            this.emotionalState.momentum.aggression = recent.reduce((s, e) => s + e.aggression, 0) / recentFrames;
            this.emotionalState.momentum.frustration = recent.reduce((s, e) => s + e.frustration, 0) / recentFrames;
            this.emotionalState.momentum.desperation = recent.reduce((s, e) => s + e.desperation, 0) / recentFrames;
            this.emotionalState.momentum.confidence = recent.reduce((s, e) => s + e.confidence, 0) / recentFrames;
            this.emotionalState.momentum.vengeance = recent.reduce((s, e) => s + e.vengeance, 0) / recentFrames;
        }
        
        const fearBlock = (1 - this.emotions.desperation) * this.emotionalState.momentum.fear;
        this.emotions.aggression = 
            this.emotionalState.momentum.frustration * 0.6 + 
            this.emotionalState.momentum.desperation * 0.8 +
            this.emotionalState.momentum.vengeance * 0.9 -
            fearBlock * 0.4;
        
        this.S[0] = this.emotions.threat;
        this.S[1] = this.emotions.frustration;
        this.S[2] = this.emotions.confidence;
        this.S[3] = this.emotions.aggression;
        this.S[4] = this.emotions.fear;
        this.S[5] = this.emotions.surprise;
        this.S[6] = this.emotions.caution;
        this.S[7] = this.emotions.desperation;
        this.S[8] = this.emotions.vengeance;
        
        for (const key of Object.keys(this.emotions)) {
            this.emotions[key] *= this.emotionDecay;
        }
        this._lastReward = r_t;
    }
    
    _getDominantEmotion() {
        const emotions = [
            { name: 'vengeance', value: this.emotions.vengeance },
            { name: 'desperation', value: this.emotions.desperation },
            { name: 'fear', value: this.emotions.fear },            { name: 'aggression', value: this.emotions.aggression },
            { name: 'frustration', value: this.emotions.frustration },
            { name: 'confidence', value: this.emotions.confidence }
        ];
        emotions.sort((a, b) => b.value - a.value);
        return { name: emotions[0].name, intensity: emotions[0].value };
    }
    
    _enterEmotion(name, intensity) {
        const config = this.emotionConfig[name];
        if (!config) return;
        if (this.emotionalState.current) {
            this.emotionalState.history.push({ ...this.emotionalState.current, endStep: this.step, reason: 'transition' });
            if (this.emotionalState.history.length > 5) this.emotionalState.history.shift();
        }
        this.emotionalState.current = {
            name, intensity, initialIntensity: intensity,
            baseDuration: config.baseDuration,
            remainingFrames: Math.floor(config.baseDuration * intensity * config.intensityMultiplier),
            satisfaction: this.emotionalState.satisfaction,
            startStep: this.step
        };
    }
    
    _exitEmotion(state, reason = 'completed') {
        this.emotionalState.history.push({ ...state, endStep: this.step, reason });
        if (this.emotionalState.history.length > 5) this.emotionalState.history.shift();
        this.emotionMemory.lastEmotion = {
            name: state.name, intensity: state.intensity,
            satisfaction: state.satisfaction,
            completed: reason === 'completed' || reason === 'satisfied'
        };
        if (reason === 'transition' || reason === 'interrupted') {
            this.emotionMemory.unresolved = this._clamp(this.emotionMemory.unresolved + state.intensity * 0.5, 0, 1);
        }
        this.emotionalState.current = null;
    }
    
    _canExitEmotion(state) {
        if (state.remainingFrames < state.baseDuration * 0.3) return false;
        if (state.satisfaction > 0.6) return true;
        if (state.intensity < 0.3) return true;
        return false;
    }
    
    _shouldExtendEmotion(state) {
        if (state.intensity > state.initialIntensity + 0.2) return true;
        if (state.satisfaction < -0.3) return true;
        return false;
    }    
    _checkEmotionTransition(state) {
        const current = state.name;
        if (current === 'fear' && this.emotions.desperation > 0.7) return { name: 'desperation', intensity: this.emotions.desperation };
        if (current === 'frustration' && this.emotions.aggression > 0.6) return { name: 'aggression', intensity: this.emotions.aggression };
        if (current === 'aggression' && state.satisfaction < -0.5) return { name: 'frustration', intensity: this.emotions.frustration };
        if (current === 'desperation' && state.intensity < 0.4) return { name: 'fear', intensity: this.emotions.fear };
        if (current === 'vengeance' && state.satisfaction > 0.5) return { name: 'confidence', intensity: this.emotions.confidence };
        return null;
    }
    
    _processCombatEvents(events) {
        if (!events) return;
        for (const event of events) {
            switch(event.type) {
                case 'hit_dealt':
                    this.emotionalState.satisfaction = this._clamp(this.emotionalState.satisfaction + 0.15, -1, 1);
                    if (this.emotionalState.current?.name === 'aggression' || this.emotionalState.current?.name === 'vengeance') {
                        this.emotionalState.current.intensity = Math.min(1, this.emotionalState.current.intensity + 0.05);
                    }
                    break;
                case 'hit_missed':
                    this.emotions.frustration = Math.min(1, this.emotions.frustration + 0.1);
                    this.emotionalState.satisfaction = this._clamp(this.emotionalState.satisfaction - 0.1, -1, 1);
                    break;
                case 'damage_received':
                    this.emotions.fear = Math.min(1, this.emotions.fear + (event.damage || 1) * 0.02);
                    this.emotionalState.satisfaction = this._clamp(this.emotionalState.satisfaction - 0.2, -1, 1);
                    if (this.health / this.stats.maxHealth < 0.3) this.emotions.desperation = Math.min(1, this.emotions.desperation + 0.15);
                    break;
                case 'enemy_defeated':
                    this.emotions.confidence = Math.min(1, this.emotions.confidence + 0.3);
                    this.emotionalState.satisfaction = this._clamp(this.emotionalState.satisfaction + 0.5, -1, 1);
                    if (this.emotionalState.current?.name === 'aggression' || this.emotionalState.current?.name === 'vengeance') {
                        this._exitEmotion(this.emotionalState.current, 'satisfied');
                    }
                    break;
                case 'killed_by':
                    if (event.killer) {
                        this.emotionMemory.killer = event.killer;
                        const grudge = this.emotionMemory.grudge.get(event.killer) || 0;
                        this.emotionMemory.grudge.set(event.killer, this._clamp(grudge + 0.5, -1, 1));
                        this.emotions.vengeance = Math.min(1, this.emotions.vengeance + 0.6);
                    }
                    break;
            }
        }
    }
    
    _checkReflexPath(x_t, S_new, D_new) {        const emotionModulator = this.emotionInfluence;
        if (S_new[8] > 0.6) {
            this._stats.reflexTriggers++;
            D_new[6] = this._lerp(D_new[6], Math.min(1, D_new[6] + 0.4), emotionModulator);
            D_new[4] = this._lerp(D_new[4], Math.min(1, D_new[4] + 0.3), emotionModulator);
        }
        if (S_new[7] > 0.6) {
            this._stats.reflexTriggers++;
            D_new[6] = this._lerp(D_new[6], Math.min(1, D_new[6] + 0.35), emotionModulator);
            D_new[4] = this._lerp(D_new[4], Math.min(1, D_new[4] + 0.3), emotionModulator);
        }
        if (S_new[4] > 0.7) {
            this._stats.reflexTriggers++;
            D_new[10] = this._lerp(D_new[10], Math.min(1, D_new[10] + 0.35), emotionModulator);
            D_new[4] = this._lerp(D_new[4], Math.max(0, D_new[4] - 0.2), emotionModulator);
        }
        if (S_new[1] > 0.7) {
            this._stats.reflexTriggers++;
            D_new[6] = this._lerp(D_new[6], Math.min(1, D_new[6] + 0.3), emotionModulator);
            D_new[4] = this._lerp(D_new[4], Math.min(1, D_new[4] + 0.25), emotionModulator);
        }
    }
    
    _resolveCommandConflicts(D_new, emotions) {
        const emotionBias = this.emotionInfluence;
        if (emotions.vengeance > 0.6 || emotions.desperation > 0.6) {
            D_new[10] = Math.max(0, D_new[10] - emotionBias);
            D_new[7] = Math.min(1, D_new[7] + emotionBias * 0.5);
        }
        if (emotions.fear > 0.7) {
            D_new[6] = Math.max(0, D_new[6] - emotionBias);
            D_new[4] = Math.max(0, D_new[4] - emotionBias * 0.3);
        }
        if (emotions.aggression > 0.6) {
            D_new[6] = Math.min(1, D_new[6] + emotionBias);
            D_new[4] = Math.min(1, D_new[4] + emotionBias * 0.3);
        }
    }
    
    _processSocialInput(socialContext) {
        const others = socialContext.others || [];
        for (const other of others) {
            const rel = this.socialMemory.relationships.get(other.id) || 
                       { trust: 0, affinity: 0, lastInteraction: 0, sharedHistory: [] };
            const otherHealthRatio = other.healthRatio || 1.0;
            const otherIsThreatened = otherHealthRatio < 0.4;
            const otherIsAggressive = other.aggression > 0.7;
            
            if (otherIsThreatened && this.personality.traits.empathy > 0.6) {
                this.emotions.empathy = Math.min(1, this.emotions.empathy + 0.1);            }
            if (otherIsAggressive && rel.trust < 0) {
                this.emotions.fear = Math.min(1, this.emotions.fear + 0.15);
            }
            
            const dist = other.distance || 100;
            const recentInteraction = dist < 100;
            if (recentInteraction) {
                if (!otherIsAggressive && dist < 80) {
                    rel.trust = this._clamp(rel.trust + 0.05, -1, 1);
                    rel.affinity = this._clamp(rel.affinity + 0.03, -1, 1);
                } else if (otherIsAggressive) {
                    rel.trust = this._clamp(rel.trust - 0.1, -1, 1);
                    rel.affinity = this._clamp(rel.affinity - 0.05, -1, 1);
                }
                rel.lastInteraction = this.step;
                if (rel.sharedHistory.length > 20) rel.sharedHistory.shift();
            }
            rel.trust *= 0.999;
            rel.affinity *= 0.999;
            this.socialMemory.relationships.set(other.id, rel);
        }
    }
    
    _recallRelevantMemories(currentContext) {
        const { enemyId, healthRatio, situation } = currentContext;
        const recalled = [];
        for (const event of this.episodicMemory.events) {
            if (enemyId && event.agentId === enemyId) {
                recalled.push({ ...event, relevance: event.importance * 1.5 });
            }
            if (situation === 'low_health' && event.emotion === 'fear') {
                recalled.push({ ...event, relevance: event.importance * 1.2 });
            }
            if (situation === 'advantage' && event.emotion === 'pride') {
                recalled.push({ ...event, relevance: event.importance * 1.2 });
            }
        }
        recalled.sort((a, b) => b.relevance - a.relevance);
        return recalled.slice(0, 3);
    }
}

// ============================================================================
// SEEDED RNG (for reproducibility)
// ============================================================================

class SeededRNG {
    constructor(seed) {
        this._baseSeed = String(seed || 'default').split('').reduce((a, c, i) => {            return ((a * 31 + c.charCodeAt(0)) ^ (i * 17)) % 2147483647;
        }, String(seed || 'default').length) || 12345;
        this._state = this._baseSeed;
    }
    
    next() {
        let x = this._state;
        x ^= x << 13;
        x ^= x >>> 17;
        x ^= x << 5;
        this._state = x >>> 0;
        return (x * 0x2545F4914F6CDD1D) >>> 0 / 4294967296;
    }
    
    gaussian(mean = 0, std = 1) {
        const u1 = this.next();
        const u2 = this.next();
        const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
        return z * std + mean;
    }
}

// ============================================================================
// CROSS-PLATFORM EXPORT
// ============================================================================

// Node.js / CommonJS
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { Cortex2Brain, SeededRNG };
}

// Browser global
if (typeof window !== 'undefined') {
    window.Cortex2Brain = Cortex2Brain;
    window.SeededRNG = SeededRNG;
}

// Web Worker
if (typeof self !== 'undefined' && typeof self.postMessage === 'function') {
    self.Cortex2Brain = Cortex2Brain;
    self.SeededRNG = SeededRNG;
}

// ES Module (default)
export { Cortex2Brain, SeededRNG };
