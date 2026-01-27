"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.EvolutionIntegration = void 0;
const BaseIntegration_1 = require("../base/BaseIntegration");
class EvolutionIntegration extends BaseIntegration_1.BaseIntegration {
    constructor(client) {
        super(client, '/api/v1/evolution');
        this.name = 'evolution';
        this.version = '1.0.0';
        this.description = 'Evolutionary algorithms and adversarial testing';
    }
    async execute(inputs) {
        const validation = this.validate(inputs);
        if (!validation.valid) {
            throw new Error(`Invalid inputs: ${validation.errors.map(e => e.message).join(', ')}`);
        }
        return this.request('POST', inputs);
    }
    getSchema() {
        return {
            type: 'object',
            properties: {
                mode: {
                    type: 'string',
                    description: 'Evolution mode',
                    enum: ['evolutionary', 'adversarial'],
                    default: 'evolutionary'
                },
                initial_population: {
                    type: 'array',
                    description: 'Initial population of solutions',
                    items: {}
                },
                fitness_function: {
                    type: 'string',
                    description: 'Fitness function to evaluate solutions'
                },
                generations: {
                    type: 'number',
                    description: 'Number of generations to evolve',
                    minimum: 1,
                    maximum: 10000,
                    default: 100
                },
                mutation_rate: {
                    type: 'number',
                    description: 'Mutation rate (0-1)',
                    minimum: 0,
                    maximum: 1,
                    default: 0.1
                },
                crossover_rate: {
                    type: 'number',
                    description: 'Crossover rate (0-1)',
                    minimum: 0,
                    maximum: 1,
                    default: 0.8
                },
                selection_method: {
                    type: 'string',
                    description: 'Selection method',
                    enum: ['tournament', 'roulette', 'rank'],
                    default: 'tournament'
                }
            },
            required: ['initial_population', 'fitness_function', 'generations']
        };
    }
    async evolve(population, generations) {
        return this.execute({
            mode: 'evolutionary',
            initial_population: population,
            fitness_function: 'maximize_accuracy',
            generations
        });
    }
    async evolveAdversarial(baseSolution, attackStrategies = ['fgsm', 'pgd']) {
        return this.execute({
            mode: 'adversarial',
            initial_population: [baseSolution],
            fitness_function: 'adversarial_robustness',
            generations: 50
        });
    }
}
exports.EvolutionIntegration = EvolutionIntegration;
//# sourceMappingURL=evolution.js.map