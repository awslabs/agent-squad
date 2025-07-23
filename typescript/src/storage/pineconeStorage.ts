import { Pinecone} from '@pinecone-database/pinecone';
import { ChatHistory, ConversationMessage, ParticipantRole } from '../types';
import { ChatStorage } from './chatStorage';
import { VectorUtils } from '../utils/vectorUtils';
import { Logger } from '../utils/logger';
import { LLMUtils } from '../utils/llmUtils';
import { v4 as uuidv4 } from 'uuid';
const VECTOR_DIMENSION = 1024;
export interface PineconeChatMessage {
  id: string;
  userId: string;
  channelId?: string;
  threadTs?: string;
  content: string;
  timestamp: number;
  messageType: 'user' | 'bot';
  metadata?: Record<string, any>;
  isDM: boolean;
}

export class PineconeStorage extends ChatStorage {
  private client: Pinecone;
  private indexName: string;
  private vectorUtils: VectorUtils;
  private llmUtils: LLMUtils;
  private maxCount: number;

  constructor(indexName: string, llmUtils: LLMUtils, maxCount: number) {
    super();
    this.client = new Pinecone({
      apiKey: process.env['PINECONE_KEY'],
    });
    this.indexName = indexName || process.env.PINECONE_INDEX;
    this.vectorUtils = new VectorUtils();
    this.llmUtils = llmUtils;
    this.maxCount = maxCount || 20;
  }

  async saveChatMessage(
    userId: string,
    sessionId: string,
    agentId: string,
    newMessage: ConversationMessage
  ): Promise<ConversationMessage[]> {
    const [channelId, threadTs] = this.parseSessionId(sessionId);
    const message: PineconeChatMessage = {
      id: this.generatePineconeId(channelId, threadTs, userId),
      userId,
      channelId,
      threadTs,
      content: newMessage.content[0].text,
      timestamp: Date.now(),
      messageType: newMessage.role === 'user' ? 'user' : 'bot',
      metadata: { agentId, sessionId, channelId, threadTs },
      isDM: userId === sessionId
    };

    await this.storeMessage(message);
    return [newMessage];
  }

  async fetchChat(userId: string, sessionId: string): Promise<ConversationMessage[]> {
    const [channelId, threadTs] = this.parseSessionId(sessionId);
    const id = this.generatePineconeId(channelId, threadTs, userId);

    // Fetch vector (by id) from Pinecone
    const result = await this.client.index(this.indexName).fetch([id]);
    // const vector = result.vectors?.[id];
    const record = result.records?.[id]
    if (!record) return [];

    const metadata = record.metadata || {};
    return [{
      role: metadata.messageType === 'user' ? ParticipantRole.USER : ParticipantRole.ASSISTANT,
      content: [metadata["content"]]
    }];
  }

  async fetchAllChats(
    userId: string,
    sessionId: string,
    query: string
  ): Promise<ChatHistory> {
    const [channelId, threadTs] = this.parseSessionId(sessionId);
    Logger.logger.info(`Fetching All Chats: userid:${userId} channelid: ${channelId} threadTs:${threadTs}`);
    const context = await this.getRelevantContext(query, userId, channelId, threadTs, this.maxCount);

    const messages = context.messages.map(msg => ({
      role: msg.metadata?.messageType === 'user' ? ParticipantRole.USER : ParticipantRole.ASSISTANT,
      content: [{ text: msg.metadata?.content }],
      timestamp: msg.metadata?.timestamp,
      metadata: msg.metadata,
    }));

    return { messages, summary: context.summary };
  }

  async getRelevantContext(
    query: string,
    userId: string,
    channelId?: string,
    threadTs?: string,
    maxResults = 10
  ): Promise<{ messages: any[]; summary?: string }> {
    try {
      const contextId = this.generateContextId({
        id: '',
        userId,
        channelId,
        threadTs,
        content: query,
        timestamp: Date.now(),
        messageType: 'user',
        isDM: userId === channelId,
      });
      Logger.logger.info(`Fetching relevant context: contextId:${contextId}`);

      // Generate embedding
      const queryEmbedding = await this.vectorUtils.generateEmbedding(query);
//filter: { "$and": [{"contextId": contextId}, {"isActive": true}] },
      // Vector similarity search with filter
      const response = await this.client.index(this.indexName).query({
        vector: queryEmbedding,
        topK: Math.floor(maxResults * 0.7),
        filter: { 
            contextId: {$eq: contextId},
            isActive: {$eq: true}
         },
        includeMetadata: true,
        includeValues: false
      });

      const messages = response.matches?.map(match => ({
        id: match.id,
        metadata: match.metadata
      })) ?? [];

      // Add recent fallback messages
      const recentMessages = await this.getRecentMessages(contextId, maxResults - messages.length);
      const combined = [...messages, ...recentMessages];
      // Deduplicate by id
      const seen = new Set();
      const uniqueMessages = combined.filter(m => {
        if (seen.has(m.id)) return false;
        seen.add(m.id);
        return true;
      });

      // Sort by timestamp
      uniqueMessages.sort((a, b) => (a.metadata?.timestamp || 0) - (b.metadata?.timestamp || 0));

      // Get summary
      const summary = await this.getContextSummary(contextId);

      return {
        messages: uniqueMessages.slice(-maxResults),
        summary
      };
    } catch (error) {
      console.error('Error getting relevant context:', error);
      throw error;
    }
  }

  async getRecentMessages(contextId: string, limit = 20): Promise<any[]> {
    // Pinecone doesn't have a scroll; instead use filtered query for recent active messages.
    Logger.logger.info('getting recent messages');
    const response = await this.client.index(this.indexName).query({
      vector: Array(VECTOR_DIMENSION).fill(0), // "Zero" vector to get recent; you might use a random probe or track recency in metadata.
      topK: limit,
      filter: { 
        contextId: {$eq: contextId},
        isActive: {$eq: true},
     },
      includeMetadata: true
    });
    return response.matches?.map(match => ({
      id: match.id,
      metadata: match.metadata
    })) ?? [];
  }

  async storeMessage(message: PineconeChatMessage): Promise<void> {
    try {
      const contextId = this.generateContextId(message);
      const contextType = this.getContextType(message);

      const embedding = await this.vectorUtils.generateEmbedding(message.content);

      // Get the current message count for index
      const recents = await this.getRecentMessages(contextId, 1000);
      const messageIndex = recents.length;

      Logger.logger.info(`Saving to vector store with id: ${message.id}`);
      // Upsert (Insert/update) into Pinecone index
      await this.client.index(this.indexName).upsert([
        {
          id: message.id,
          values: embedding,
          metadata: {
            contextId,
            contextType,
            userId: message.userId,
            channelId: message.channelId,
            threadTs: message.threadTs,
            content: message.content,
            messageType: message.messageType,
            timestamp: message.timestamp,
            messageIndex,
            isActive: true,
            ...message.metadata,
          }
        }
      ]);

      // Prune context to max N active messages
      const maxActiveMessages = contextType === 'channel_thread' ? 100 : 200;
      await this.deactivateOldMessages(contextId, maxActiveMessages);

      // Trigger summary every 50th message
      if (messageIndex > 0 && messageIndex % 50 === 0) {
        await this.createContextSummary(contextId);
      }
      Logger.logger.info('Saved to vectorstore');
    } catch (error) {
      Logger.logger.error('Error storing message:', error);
      throw error;
    }
  }

  async deactivateOldMessages(contextId: string, keepRecent = 50): Promise<void> {
    const messages = await this.getRecentMessages(contextId, 1000);
    if (messages.length <= keepRecent) return;
    const toDeactivate = messages.slice(keepRecent);
    Logger.logger.info('Deactivating old messages');
    for (const record of toDeactivate) {
      await this.client.index(this.indexName).upsert([
        {
          id: record.id,
          values: record.values ?? Array(VECTOR_DIMENSION).fill(0),  // Pinecone requires values -- you may need to cache/fetch these!
          metadata: { ...record.metadata, isActive: false }
        }
      ]);
    }
  }

  private async createContextSummary(contextId: string): Promise<void> {
    // Use getRecentMessages() to collect messages and summarize via LLM
    const messages = await this.getRecentMessages(contextId, 50);

    if (messages.length < 10) return;

    const conversationText = messages
      .sort((a, b) => (a.metadata?.timestamp ?? 0) - (b.metadata?.timestamp ?? 0))
      .map(msg => `${msg.metadata?.messageType}: ${msg.metadata?.content}`)
      .join('\n');

    const summary = await this.llmUtils.generateSummary(conversationText);
    const summaryEmbedding = await this.vectorUtils.generateEmbedding(summary);

    Logger.logger.info('Saving summary to vector store');
    await this.client.index(this.indexName).upsert([
      {
        id: uuidv4(),
        values: summaryEmbedding,
        metadata: {
          contextId,
          contextType: contextId.startsWith('channel_') ? 'channel_thread' : 'user_dm',
          userId: 'system',
          content: summary,
          messageType: 'bot',
          timestamp: Date.now(),
          messageIndex: -1,
          isActive: true,
          type: 'summary'
        }
      }
    ]);
  }

  private parseSessionId(
    sessionId: string
  ): [string | undefined, string | undefined] {
    const parts = sessionId.split("#");
    if (parts.length === 2) {
      return [parts[0], parts[1]];
    }
    return [sessionId, sessionId];
  }

  private getContextType(message: PineconeChatMessage): "channel_thread" | "user_dm" {
    return !message.isDM ? "channel_thread" : "user_dm";
  }

  private generateContextId(message: PineconeChatMessage): string {
    if (message.isDM) {
      return `dm_${message.userId}`;
    } else {
      return `channel_${message.channelId}_thread_${message.threadTs}`;
    }
  }

  private formContextId(userId, channelId, threadTs) {
    if (userId === channelId) {
      return `dm_${userId}`;
    } else {
      return `channel_${channelId}_thread_${threadTs}`;
    }
  }

  private generatePineconeId(channelId, threadTs, userId){
    const uniquePart = uuidv4().replace(/-/g, '').slice(0, 8); // 8-character UUID segment
    return `${this.formContextId(userId, channelId, threadTs)}_${uniquePart}`
  }

  private async getContextSummary(contextId: string): Promise<string | undefined> {
    // Pinecone does not support attribute ordering. Instead, perform filtered query for summaries.
    const response = await this.client.index(this.indexName).query({
      vector: Array(VECTOR_DIMENSION).fill(0),
      topK: 1,
      filter: { 
        contextId: {$eq: contextId},
        isActive: {$eq: true},
        type: {$eq: "summary"}
     },
      includeMetadata: true
    });
    const match = response.matches?.[0];
    return match?.metadata?.content.toString();
  }
}
