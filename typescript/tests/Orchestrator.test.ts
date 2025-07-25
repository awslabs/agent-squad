import { AgentSquad, OrchestratorOptions } from '../src/orchestrator';
import { Agent, AgentOptions } from '../src/agents/agent';
import { BedrockClassifier } from '../src/classifiers/bedrockClassifier';
import { InMemoryChatStorage } from '../src/storage/memoryChatStorage';
import { ParticipantRole } from '../src/types/index';
import { ClassifierResult } from '../src/classifiers/classifier';
import { ConversationMessage } from '../src/types/index';
import { AccumulatorTransform } from '../src/utils/helpers';
import * as chatUtils from '../src/utils/chatUtils';

// Mock the dependencies
jest.mock('../src/classifiers/bedrockClassifier');
jest.mock('../src/storage/memoryChatStorage');
jest.mock('../src/utils/helpers');
jest.mock('../src/utils/chatUtils', () => ({
  saveConversationExchange: jest.fn(),
}));


// Create a mock Agent class
class MockAgent extends Agent {
    constructor(options: AgentOptions) {
      super(options);
    }

    async processRequest(
      _inputText: string,
      _userId: string,
      _sessionId: string,
      _chatHistory: ConversationMessage[],
      _additionalParams?: Record<string, string>
    ): Promise<ConversationMessage | AsyncIterable<any>> {
      // This is a mock implementation
      return {
        role: ParticipantRole.ASSISTANT,
        content: [{ text: 'Mock response' }],
      };
    }
  }


describe('AgentSquad', () => {


  let orchestrator: AgentSquad;
  let mockAgent: Agent;
  let mockClassifier: jest.Mocked<BedrockClassifier>;
  let mockStorage: jest.Mocked<InMemoryChatStorage>;

  beforeEach(() => {
    mockAgent = new MockAgent({
      name: 'Test Agent',
      description: 'A test agent'
    });

    mockClassifier = new BedrockClassifier() as jest.Mocked<BedrockClassifier>;
    mockStorage = new InMemoryChatStorage() as jest.Mocked<InMemoryChatStorage>;

    // Configure orchestrator options
    const options: OrchestratorOptions = {
      storage: mockStorage,
      classifier: mockClassifier,
      config: {
        LOG_EXECUTION_TIMES: true,
        USE_DEFAULT_AGENT_IF_NONE_IDENTIFIED: true,
      },
    };

    // Create orchestrator instance
    orchestrator = new AgentSquad(options);
    orchestrator.addAgent(mockAgent);
  });

  test('addAgent adds an agent and updates classifier', () => {

    const newAgent: Agent = {
      id: 'new-agent',
      name: 'New Agent',
      description: 'A new test agent',
      processRequest: jest.fn(),
    } as unknown as jest.Mocked<Agent>;

    orchestrator.addAgent(newAgent);
    expect(mockClassifier.setAgents).toHaveBeenCalled();
  });

  test('getAllAgents returns all added agents', () => {

    const agents = orchestrator.getAllAgents();

    expect(agents).toHaveProperty('test-agent');
    expect(agents['test-agent']).toEqual({
      name: 'Test Agent',
      description: 'A test agent',
    });
  });

  test('routeRequest with identified agent', async () => {

    const processRequestSpy = jest.spyOn(mockAgent, 'processRequest');
    const userInput = 'Test input';
    const userId = 'user1';
    const sessionId = 'session1';

    const mockClassifierResult: ClassifierResult = {
      selectedAgent: mockAgent,
      confidence: 0.8,
    };

    mockClassifier.classify.mockResolvedValue(mockClassifierResult);

    mockStorage.fetchAllChats.mockResolvedValue([]);

    const response = await orchestrator.routeRequest(userInput, userId, sessionId);

    expect(response.output).toBe('Mock response');
    expect(response.metadata.agentId).toBe(mockAgent.id);
    expect(mockStorage.fetchAllChats).toHaveBeenCalledWith(userId, sessionId);
    expect(mockClassifier.classify).toHaveBeenCalledWith(userInput, []);
    expect(processRequestSpy).toHaveBeenCalled();
  });


  test('routeRequest with streaming response', async () => {
    const userInput = 'Stream input';
    const userId = 'user1';
    const sessionId = 'session1';

    const mockAgent = {
      id: 'test-agent',
      name: 'Test Agent',
      description: 'A test agent',
      processRequest: jest.fn(),
    } as unknown as jest.Mocked<Agent>;

    const mockClassifierResult: ClassifierResult = {
      selectedAgent: mockAgent,
      confidence: 0.8,
    };

    mockClassifier.classify.mockResolvedValue(mockClassifierResult);

    const mockStream = (async function* () {
      yield 'Chunk 1';
      yield 'Chunk 2';
    })();

    mockAgent.processRequest.mockResolvedValue(mockStream);

    const mockAccumulatorTransform = new AccumulatorTransform();
    (AccumulatorTransform as jest.MockedClass<typeof AccumulatorTransform>).mockImplementation(() => mockAccumulatorTransform);

    const response = await orchestrator.routeRequest(userInput, userId, sessionId);

    expect(response.streaming).toBe(true);
    expect(response.output).toBe(mockAccumulatorTransform);
  });

  test('streaming response saves conversation history with maxHistorySize', async () => {
    // Create a custom orchestrator with specific MAX_MESSAGE_PAIRS_PER_AGENT
    const customOrchestrator = new AgentSquad({
      storage: mockStorage,
      classifier: mockClassifier,
      config: {
        LOG_EXECUTION_TIMES: true,
        USE_DEFAULT_AGENT_IF_NONE_IDENTIFIED: true,
        MAX_MESSAGE_PAIRS_PER_AGENT: 10,
      },
    });

    const userInput = 'Test streaming save';
    const userId = 'user1';
    const sessionId = 'session1';

    const mockAgent = {
      id: 'test-agent',
      name: 'Test Agent',
      description: 'A test agent',
      processRequest: jest.fn(),
      saveChat: true,
    } as unknown as jest.Mocked<Agent>;

    customOrchestrator.addAgent(mockAgent);

    const mockClassifierResult: ClassifierResult = {
      selectedAgent: mockAgent,
      confidence: 0.9,
    };

    mockClassifier.classify.mockResolvedValue(mockClassifierResult);

    const mockStream = (async function* () {
      yield 'Hello';
      yield ' world';
      yield '!';
    })();

    mockAgent.processRequest.mockResolvedValue(mockStream);

    // Mock the AccumulatorTransform to simulate streaming completion
    const mockAccumulatorTransform = {
      getAccumulatedData: jest.fn().mockReturnValue('Hello world!'),
      on: jest.fn().mockImplementation((event, handler) => {
        if (event === 'finish') {
          // Simulate stream finishing after a short delay
          setTimeout(() => handler(), 10);
        }
        return mockAccumulatorTransform;
      }),
      write: jest.fn().mockReturnValue(true),
      end: jest.fn(),
      destroy: jest.fn(),
      pipe: jest.fn(),
      unpipe: jest.fn(),
      removeListener: jest.fn(),
    };
    
    (AccumulatorTransform as jest.MockedClass<typeof AccumulatorTransform>).mockImplementation(() => mockAccumulatorTransform as any);

    // Mock storage methods
    mockStorage.fetchAllChats.mockResolvedValue([]);

    const response = await customOrchestrator.routeRequest(userInput, userId, sessionId);

    // Wait for the stream to finish processing
    await new Promise(resolve => setTimeout(resolve, 50));

    // Verify that saveConversationExchange was called with the correct parameters
    expect(chatUtils.saveConversationExchange).toHaveBeenCalledWith(
      userInput,
      'Hello world!',
      mockStorage,
      userId,
      sessionId,
      mockAgent.id,
      10 // MAX_MESSAGE_PAIRS_PER_AGENT from config
    );
  });

  test('setDefaultAgent changes the default agent', () => {
    const newDefaultAgent: Agent = {
      id: 'new-default',
      name: 'New Default Agent',
      description: 'A new default agent',
      processRequest: jest.fn(),
    } as unknown as jest.Mocked<Agent>;

    orchestrator.setDefaultAgent(newDefaultAgent);
    expect(orchestrator.getDefaultAgent()).toBe(newDefaultAgent);
  });


  test('addAgent throws error when adding an agent with an existing ID', () => {
    const existingAgent: Agent = {
      id: 'existing-agent',
      name: 'Existing Agent',
      description: 'An existing test agent',
      processRequest: jest.fn(),
    } as unknown as jest.Mocked<Agent>;

    // Add the existing agent
    orchestrator.addAgent(existingAgent);

    // Attempt to add an agent with the same ID
    const duplicateAgent: Agent = {
      id: 'existing-agent',
      name: 'Duplicate Agent',
      description: 'A duplicate test agent',
      processRequest: jest.fn(),
    } as unknown as jest.Mocked<Agent>;

    // Expect an error to be thrown when adding the duplicate agent
    expect(() => {
      orchestrator.addAgent(duplicateAgent);
    }).toThrow(`An agent with ID 'existing-agent' already exists.`);

    // Verify that the classifier's setAgents method was only called once (for the first agent)
    expect(mockClassifier.setAgents).toHaveBeenCalledTimes(2); // Once in beforeEach, once for existingAgent
  });


});

describe('AgentSquad saveConversationExchange', () => {
  let orchestrator: AgentSquad;
  let mockAgent: MockAgent;
  let mockClassifier: jest.Mocked<BedrockClassifier>;
  let mockStorage: jest.Mocked<InMemoryChatStorage>;

  beforeEach(() => {
    jest.clearAllMocks();

    mockAgent = new MockAgent({
      name: 'Mock Agent',
      description: 'A mock agent',
      saveChat: true
    });

    mockClassifier = new BedrockClassifier() as jest.Mocked<BedrockClassifier>;
    mockStorage = new InMemoryChatStorage() as jest.Mocked<InMemoryChatStorage>;

    const options: OrchestratorOptions = {
      storage: mockStorage,
      classifier: mockClassifier
    };

    orchestrator = new AgentSquad(options);
    orchestrator.addAgent(mockAgent);
  });

  test('routeRequest calls saveConversationExchange for default saveChat', async () => {

    const mockAgent = new MockAgent({
      name: 'Mock Agent',
      description: 'A mock agent'
    });


    const mockClassifierResult: ClassifierResult = {
     selectedAgent: mockAgent,
      confidence: 0.5,
    };

    mockClassifier.classify.mockResolvedValue(mockClassifierResult);
    mockStorage.fetchAllChats.mockResolvedValue([]);

    jest.spyOn(orchestrator, 'dispatchToAgent').mockResolvedValue('Mock agent response');

    await orchestrator.routeRequest('Test input', 'user1', 'session1');

    expect(chatUtils.saveConversationExchange).toHaveBeenCalledWith(
      'Test input',
      'Mock agent response',
      mockStorage,
      'user1',
      'session1',
      'mock-agent',
      expect.any(Number)
    );
  });


  test('routeRequest calls saveConversationExchange for saveChat=true', async () => {

    const mockAgent = new MockAgent({
      name: 'Mock Agent',
      description: 'A mock agent',
      saveChat: true
    });


    const mockClassifierResult: ClassifierResult = {
     selectedAgent: mockAgent,
      confidence: 0.5,
    };

    mockClassifier.classify.mockResolvedValue(mockClassifierResult);
    mockStorage.fetchAllChats.mockResolvedValue([]);

    jest.spyOn(orchestrator, 'dispatchToAgent').mockResolvedValue('Mock agent response');

    await orchestrator.routeRequest('Test input', 'user1', 'session1');

    expect(chatUtils.saveConversationExchange).toHaveBeenCalledWith(
      'Test input',
      'Mock agent response',
      mockStorage,
      'user1',
      'session1',
      'mock-agent',
      expect.any(Number)
    );
  });

  test('routeRequest do not calls saveConversationExchange for saveChat=false', async () => {

    const mockAgent = new MockAgent({
      name: 'Mock Agent',
      description: 'A mock agent',
      saveChat: false
    });


    const mockClassifierResult: ClassifierResult = {
     selectedAgent: mockAgent,
      confidence: 0.5,
    };

    mockClassifier.classify.mockResolvedValue(mockClassifierResult);
    mockStorage.fetchAllChats.mockResolvedValue([]);

    jest.spyOn(orchestrator, 'dispatchToAgent').mockResolvedValue('Mock agent response');

    await orchestrator.routeRequest('Test input', 'user1', 'session1');

    expect(chatUtils.saveConversationExchange).not.toHaveBeenCalled();
  });




});