// import { TfIdf } from "natural";
// import { removeStopwords } from "stopword";
// import { Logger } from "./utils/logger";

// export interface OverlapResult {
//   overlapPercentage: string;
//   potentialConflict: "High" | "Medium" | "Low";
// }

// export interface UniquenessScore {
//   agent: string;
//   uniquenessScore: string;
// }

// export interface AnalysisResult {
//   pairwiseOverlap: { [key: string]: OverlapResult };
//   uniquenessScores: UniquenessScore[];
// }

// export class AgentOverlapAnalyzer {
//   private agents: { [key: string]: { name: string; description: string } };

//   constructor(agents: {
//     [key: string]: { name: string; description: string };
//   }) {
//     this.agents = agents;
//   }

//   analyzeOverlap(): void {
//     const agentDescriptions = Object.entries(this.agents).map(
//       ([_, agent]) => agent.description
//     );
//     const agentNames = Object.entries(this.agents).map(([key, _]) => key);

//     if (agentNames.length < 2) {
//       Logger.logger.info("Agent Overlap Analysis requires at least two agents.");
//       Logger.logger.info(`Current number of agents: ${agentNames.length}`);
//       if (agentNames.length === 1) {
//         Logger.logger.info(`\nSingle Agent Information:`);
//         Logger.logger.info(`Agent Name: ${agentNames[0]}`);
//         Logger.logger.info(`Description: ${agentDescriptions[0]}`);
//       }
//       return;
//     }

    
//     const tfidf = new TfIdf();

//     // Preprocess descriptions and add to TF-IDF
//     const _preprocessedDescriptions = agentDescriptions.map((description) => {
//       const tokens = removeStopwords(description.toLowerCase().split(/\W+/));
//       tfidf.addDocument(tokens);
//       return tokens;
//     });

//     const overlapResults: { [key: string]: OverlapResult } = {};
//     for (let i = 0; i < agentDescriptions.length; i++) {
//       for (let j = i + 1; j < agentDescriptions.length; j++) {
//         const agent1 = agentNames[i];
//         const agent2 = agentNames[j];
//         const similarity = this.calculateCosineSimilarity(
//           tfidf.listTerms(i),
//           tfidf.listTerms(j)
//         );
//         const overlapPercentage = (similarity * 100).toFixed(2);
//         const key = `${agent1}__${agent2}`;
//         overlapResults[key] = {
//           overlapPercentage: `${overlapPercentage}%`,
//           potentialConflict:
//             similarity > 0.3 ? "High" : similarity > 0.1 ? "Medium" : "Low",
//         };
//       }
//     }

//     // Calculate uniqueness scores
//     const uniquenessScores: UniquenessScore[] = agentDescriptions.map(
//       (description, index) => {
//         const otherDescriptions = agentDescriptions.filter(
//           (_, i) => i !== index
//         );
//         const similarities = otherDescriptions.map((otherDescription) => {
//           const key1 = `${agentNames[index]}__${agentNames[otherDescriptions.indexOf(otherDescription)]}`;
//           const key2 = `${agentNames[otherDescriptions.indexOf(otherDescription)]}__${agentNames[index]}`;
//           const result = overlapResults[key1] || overlapResults[key2];
//           return result ? parseFloat(result.overlapPercentage) / 100 : 0;
//         });
//         const avgSimilarity =
//           similarities.reduce((sum, sim) => sum + sim, 0) / similarities.length;
//         return {
//           agent: agentNames[index],
//           uniquenessScore: ((1 - avgSimilarity) * 100).toFixed(2) + "%",
//         };
//       }
//     );

//     // Print pairwise overlap results
//     Logger.logger.info("Pairwise Overlap Results:");
//     Logger.logger.info("_________________________\n");
//     for (const key in overlapResults) {
//       const [agent1, agent2] = key.split("__");
//       const { overlapPercentage, potentialConflict } = overlapResults[key];
//       Logger.logger.info(
//         `${agent1} - ${agent2}:\n- Overlap Percentage - ${overlapPercentage}\n- Potential Conflict - ${potentialConflict}\n`
//       );
//     }
//     Logger.logger.info("");

//     // Print uniqueness scores
//     Logger.logger.info("Uniqueness Scores:");
//     Logger.logger.info("_________________\n");
//     uniquenessScores.forEach((score) => {
//       Logger.logger.info(
//         `Agent: ${score.agent}, Uniqueness Score: ${score.uniquenessScore}`
//       );
//     });
//   }

//   private calculateCosineSimilarity(
//     terms1: { term: string; tfidf: number }[],
//     terms2: { term: string; tfidf: number }[]
//   ): number {
//     const vector1: { [term: string]: number } = {};
//     const vector2: { [term: string]: number } = {};
//     terms1.forEach((term) => (vector1[term.term] = term.tfidf));
//     terms2.forEach((term) => (vector2[term.term] = term.tfidf));

//     const terms = new Set([...Object.keys(vector1), ...Object.keys(vector2)]);
//     let dotProduct = 0;
//     let magnitude1 = 0;
//     let magnitude2 = 0;

//     for (const term of terms) {
//       const v1 = vector1[term] || 0;
//       const v2 = vector2[term] || 0;
//       dotProduct += v1 * v2;
//       magnitude1 += v1 * v1;
//       magnitude2 += v2 * v2;
//     }

//     magnitude1 = Math.sqrt(magnitude1);
//     magnitude2 = Math.sqrt(magnitude2);

//     if (magnitude1 && magnitude2) {
//       return dotProduct / (magnitude1 * magnitude2);
//     } else {
//       return 0;
//     }
//   }
// }
