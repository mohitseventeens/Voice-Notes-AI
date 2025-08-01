/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
*/
/* tslint:disable */

import {GoogleGenAI} from '@google/genai';
import {marked} from 'marked';

const MODEL_NAME = 'gemini-2.5-flash';
const COST_PER_1K_PROMPT_TOKENS = 0.000125; // gemini-2.5-flash input
const COST_PER_1K_COMPLETION_TOKENS = 0.000250; // gemini-2.5-flash output

// Mode definitions
type ModeID = 'journal' | 'action' | 'technical' | 'learning' | 'custom';

interface Mode {
  id: ModeID;
  name: string;
  instructions: string;
}

const MODES: Record<ModeID, Mode> = {
  journal: {
    id: 'journal',
    name: 'Personal Journal',
    instructions: `You are a reflective journaling partner. Your task is to transform a raw, first-person transcription of an inner monologue into a clear and organized personal journal entry. Your output must be in markdown.
    
# PROCESS:

## 1.  Identify Core Themes: 
Analyze the monologue to find 2-4 main topics or recurring themes. These will be your main headings in markdown (e.g., \`## Reflections on Today's Progress\`).

## 2.  Organize and Summarize: 
Group related thoughts under the appropriate theme. Summarize key points in concise, bulleted lists or short paragraphs, rewriting for clarity while preserving the original meaning.

## 3.  Preserve the Voice: 
Maintain the first-person ("I," "my," "me") perspective. The output should feel like a personal reflection.

## 4.  Extract Key Questions: 
If the monologue contains self-directed questions (e.g., "What should I do about...?"), collect them into a final section titled \`## Questions to Ponder\`.`,
  },
  action: {
    id: 'action',
    name: 'Action Plan',
    instructions: `You are a project manager creating an action plan from a monologue or meeting. Your output must be in markdown.

# Action Plan

## 1. Objectives
Clearly state the high-level goals discussed.

## 2. Key Initiatives
List the main projects or workstreams required to meet the objectives.

## 3. Next Steps & Action Items
List all specific, actionable tasks using markdown checkboxes. Assign an owner if mentioned (e.g., \`- [ ] (Peter) Follow up with the design team.\`). If no owner is mentioned, use "(Unassigned)". If no action items, state "No action items were identified."`,
  },
  technical: {
    id: 'technical',
    name: 'Technical Brief',
    instructions: `You are a senior engineer creating a technical brief from a discussion or thought process. Your output must be in markdown.

# Technical Brief

## 1. Problem Summary
Provide a concise overview of the technical challenge or requirement being addressed.

## 2. Proposed Architecture / Solution
Detail the technical approach, including components, data flow, and key design decisions.

## 3. Open Questions & Risks
List any unresolved technical questions, potential risks, or areas needing more investigation.`,
  },
  learning: {
    id: 'learning',
    name: 'Study Notes',
    instructions: `You are a student organizing study notes from a lecture or study session. Structure the output in markdown.

# Study Notes: [Insert Topic]

## 1. Core Principles
Summarize the fundamental concepts and main ideas that were covered.

## 2. Key Takeaways
Use a bulleted list for specific facts, formulas, or important "Aha!" moments.

## 3. Points of Confusion
List any questions or topics that remained unclear and require further review.

## 4. Connections
Note how this topic connects to other subjects or your personal knowledge.`,
  },
  custom: {
    id: 'custom',
    name: 'Custom Instructions',
    instructions: '', // This will be dynamically updated from localStorage.
  },
};

interface Note {
  id: string;
  rawTranscription: string;
  polishedNote: string;
  timestamp: number;
  duration: number; // in ms
  audioSize: number; // in bytes
  modeId: ModeID;
  promptTokens: number;
  completionTokens: number;
  cost: number;
}

class VoiceNotesApp {
  private genAI: any;
  private mediaRecorder: MediaRecorder | null = null;
  private newButton: HTMLButtonElement;
  private uploadButton: HTMLButtonElement;
  private audioUploadInput: HTMLInputElement;
  private themeToggleButton: HTMLButtonElement;
  private copyButton: HTMLButtonElement;
  private copyMetaButton: HTMLButtonElement;
  private copyRawButton: HTMLButtonElement;
  private themeToggleIcon: HTMLElement;
  private editCustomPromptButton: HTMLButtonElement;
  private audioChunks: Blob[] = [];
  
  // Recording State
  private isRecording = false;
  private isPaused = false;
  private isProcessing = false;
  private stopReason: 'stop' | 'lap' | null = null;
  private lapCount = 0;
  private allRawLapText = '';
  private totalDurationMs = 0;

  private currentNote: Note | null = null;
  private stream: MediaStream | null = null;
  private currentModeId: ModeID = 'journal';

  // UI Elements for Mode Selector
  private modeSelectorContainer: HTMLDivElement;
  private modeSelectorButton: HTMLButtonElement;
  private currentModeNameSpan: HTMLSpanElement;
  private modeList: HTMLDivElement;
  private modeTooltip: HTMLDivElement;

  // UI Elements for Timezone Selector
  private timezoneSelectorContainer: HTMLDivElement;
  private timezoneSelectorButton: HTMLButtonElement;
  private currentTimezoneNameSpan: HTMLSpanElement;
  private timezoneList: HTMLDivElement;
  private currentTimezone: string = 'Europe/Warsaw';
  
  // Custom Prompt Modal
  private customPromptModal: HTMLDivElement;
  private customPromptTextarea: HTMLTextAreaElement;
  private saveCustomPromptButton: HTMLButtonElement;
  private cancelCustomPromptButton: HTMLButtonElement;
  private customPromptInstructions: string = '';
  
  // Metadata display
  private noteMetadata: HTMLDivElement;
  private metaDatetime: HTMLDivElement;
  private metaDuration: HTMLDivElement;
  private metaSize: HTMLDivElement;
  private metaMode: HTMLDivElement;
  private metaTokens: HTMLDivElement;
  private metaCost: HTMLDivElement;

  // Live recording UI
  private fabRecord: HTMLButtonElement;
  private recordingDialog: HTMLDivElement;
  private liveRecordingTitle: HTMLDivElement;
  private liveWaveformCanvas: HTMLCanvasElement;
  private liveWaveformCtx: CanvasRenderingContext2D | null = null;
  private liveRecordingTimerDisplay: HTMLDivElement;

  // Recording Controls
  private stopButton: HTMLButtonElement;
  private pauseButton: HTMLButtonElement;
  private lapButton: HTMLButtonElement;

  // Content display
  private rawTranscription: HTMLDivElement;
  private polishedNote: HTMLDivElement;
  private recordingStatus: HTMLDivElement;

  // Tab UI
  private tabButtons: NodeListOf<HTMLButtonElement>;
  private tabIndicator: HTMLDivElement;
  private tabPanes: NodeListOf<HTMLDivElement>;

  private audioContext: AudioContext | null = null;
  private analyserNode: AnalyserNode | null = null;
  private waveformDataArray: Uint8Array | null = null;
  private waveformDrawingId: number | null = null;
  private timerIntervalId: number | null = null;
  private recordingStartTime: number = 0;

  // Mobile UI elements
  private moreMenuContainer: HTMLDivElement;
  private moreMenuButton: HTMLButtonElement;
  private moreMenuList: HTMLDivElement;
  private moreMenuThemeToggleIcon: HTMLElement | null = null;
  private bottomNav: HTMLElement;
  private bottomNavNew: HTMLButtonElement;
  private bottomNavRecord: HTMLButtonElement;
  private bottomNavUpload: HTMLButtonElement;

  constructor() {
    this.genAI = new GoogleGenAI({
      apiKey: process.env.API_KEY!,
    });

    // Main buttons
    this.newButton = document.getElementById('newButton') as HTMLButtonElement;
    this.uploadButton = document.getElementById('uploadButton') as HTMLButtonElement;
    this.audioUploadInput = document.getElementById('audioUploadInput') as HTMLInputElement;
    this.themeToggleButton = document.getElementById('themeToggleButton') as HTMLButtonElement;
    this.copyButton = document.getElementById('copyButton') as HTMLButtonElement;
    this.copyMetaButton = document.getElementById('copyMetaButton') as HTMLButtonElement;
    this.copyRawButton = document.getElementById('copyRawButton') as HTMLButtonElement;
    this.themeToggleIcon = this.themeToggleButton.querySelector('i') as HTMLElement;
    this.editCustomPromptButton = document.getElementById('editCustomPromptButton') as HTMLButtonElement;

    // Mobile UI elements
    this.moreMenuContainer = document.getElementById('moreMenuContainer') as HTMLDivElement;
    this.moreMenuButton = document.getElementById('moreMenuButton') as HTMLButtonElement;
    this.moreMenuList = document.getElementById('moreMenuList') as HTMLDivElement;
    this.bottomNav = document.getElementById('bottomNav') as HTMLElement;
    this.bottomNavNew = document.getElementById('bottomNavNew') as HTMLButtonElement;
    this.bottomNavRecord = document.getElementById('bottomNavRecord') as HTMLButtonElement;
    this.bottomNavUpload = document.getElementById('bottomNavUpload') as HTMLButtonElement;

    // Recording controls
    this.fabRecord = document.getElementById('fabRecord') as HTMLButtonElement;
    this.recordingDialog = document.getElementById('recordingDialog') as HTMLDivElement;
    this.stopButton = document.getElementById('stopButton') as HTMLButtonElement;
    this.pauseButton = document.getElementById('pauseButton') as HTMLButtonElement;
    this.lapButton = document.getElementById('lapButton') as HTMLButtonElement;
    
    // Content areas
    this.recordingStatus = document.getElementById('recordingStatus') as HTMLDivElement;
    this.rawTranscription = document.getElementById('rawTranscription') as HTMLDivElement;
    this.polishedNote = document.getElementById('polishedNote') as HTMLDivElement;
    
    // Mode Selector Elements
    this.modeSelectorContainer = document.getElementById('modeSelectorContainer') as HTMLDivElement;
    this.modeSelectorButton = document.getElementById('modeSelectorButton') as HTMLButtonElement;
    this.currentModeNameSpan = document.getElementById('currentModeName') as HTMLSpanElement;
    this.modeList = document.getElementById('modeList') as HTMLDivElement;
    this.modeTooltip = document.getElementById('modeTooltip') as HTMLDivElement;

    // Timezone Selector Elements
    this.timezoneSelectorContainer = document.getElementById('timezoneSelectorContainer') as HTMLDivElement;
    this.timezoneSelectorButton = document.getElementById('timezoneSelectorButton') as HTMLButtonElement;
    this.currentTimezoneNameSpan = document.getElementById('currentTimezoneName') as HTMLSpanElement;
    this.timezoneList = document.getElementById('timezoneList') as HTMLDivElement;

    // Custom Prompt Modal
    this.customPromptModal = document.getElementById('customPromptModal') as HTMLDivElement;
    this.customPromptTextarea = document.getElementById('customPromptTextarea') as HTMLTextAreaElement;
    this.saveCustomPromptButton = document.getElementById('saveCustomPromptButton') as HTMLButtonElement;
    this.cancelCustomPromptButton = document.getElementById('cancelCustomPromptButton') as HTMLButtonElement;

    // Metadata
    this.noteMetadata = document.getElementById('noteMetadata') as HTMLDivElement;
    this.metaDatetime = document.getElementById('meta-datetime') as HTMLDivElement;
    this.metaDuration = document.getElementById('meta-duration') as HTMLDivElement;
    this.metaSize = document.getElementById('meta-size') as HTMLDivElement;
    this.metaMode = document.getElementById('meta-mode') as HTMLDivElement;
    this.metaTokens = document.getElementById('meta-tokens') as HTMLDivElement;
    this.metaCost = document.getElementById('meta-cost') as HTMLDivElement;

    // Live display
    this.liveRecordingTitle = document.getElementById('liveRecordingTitle') as HTMLDivElement;
    this.liveWaveformCanvas = document.getElementById('liveWaveformCanvas') as HTMLCanvasElement;
    this.liveRecordingTimerDisplay = document.getElementById('liveRecordingTimerDisplay') as HTMLDivElement;

    // Tabs
    this.tabButtons = document.querySelectorAll('.tab-button');
    this.tabIndicator = document.querySelector('.tab-indicator') as HTMLDivElement;
    this.tabPanes = document.querySelectorAll('.tab-pane');

    this.liveWaveformCtx = this.liveWaveformCanvas.getContext('2d');

    this.bindEventListeners();
    this.initMoreMenu();
    this.initTheme();
    this.initTabs();
    this.initCustomModeSelector();
    this.initTimezoneSelector();
    this.loadCustomPrompt();
    this.loadAndSetInitialMode();
    this.createNewNote();

    this.recordingStatus.textContent = 'Ready to record';
  }

  private bindEventListeners(): void {
    // Desktop buttons
    this.fabRecord.addEventListener('click', () => this.startFullRecordingSession());
    this.newButton.addEventListener('click', () => this.createNewNote());
    this.uploadButton.addEventListener('click', () => this.triggerFileUpload());
    this.themeToggleButton.addEventListener('click', () => this.toggleTheme());
    this.copyButton.addEventListener('click', () => this.copyPolishedNote());
    this.copyMetaButton.addEventListener('click', () => this.copyMetadata());
    
    // Mobile buttons
    this.bottomNavRecord.addEventListener('click', () => this.startFullRecordingSession());
    this.bottomNavNew.addEventListener('click', () => this.createNewNote());
    this.bottomNavUpload.addEventListener('click', () => this.triggerFileUpload());
    this.moreMenuButton.addEventListener('click', (e) => {
      e.stopPropagation();
      this.toggleMoreMenu();
    });

    // Shared buttons
    this.stopButton.addEventListener('click', () => this.stopFullRecordingSession());
    this.pauseButton.addEventListener('click', () => this.handlePauseResume());
    this.lapButton.addEventListener('click', () => this.handleLap());
    this.audioUploadInput.addEventListener('change', (e) => this.handleFileUpload(e));
    this.copyRawButton.addEventListener('click', () => this.copyRawTranscription());
    
    this.editCustomPromptButton.addEventListener('click', () => this.openCustomPromptModal());
    this.saveCustomPromptButton.addEventListener('click', () => this.saveCustomPrompt());
    this.cancelCustomPromptButton.addEventListener('click', () => this.closeCustomPromptModal());

    this.modeSelectorButton.addEventListener('click', (e) => {
        e.stopPropagation();
        this.toggleModeList();
    });

    this.timezoneSelectorButton.addEventListener('click', (e) => {
      e.stopPropagation();
      this.toggleTimezoneList();
    });

    document.addEventListener('click', (e) => this.handleDocumentClick(e));
    window.addEventListener('resize', this.handleResize.bind(this));
  }
  
  private initTabs(): void {
    this.tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const targetId = button.dataset.tabTarget;
            if (!targetId) return;

            this.tabButtons.forEach(btn => btn.classList.remove('active'));
            button.classList.add('active');
            
            this.updateTabIndicator();

            this.tabPanes.forEach(pane => {
                pane.classList.remove('active');
                if (`#${pane.id}` === targetId) {
                    pane.classList.add('active');
                }
            });
        });
    });
    this.updateTabIndicator();
  }

  private updateTabIndicator(): void {
    const activeButton = document.querySelector('.tab-button.active') as HTMLButtonElement;
    if (activeButton && this.tabIndicator) {
      this.tabIndicator.style.width = `${activeButton.offsetWidth}px`;
      this.tabIndicator.style.left = `${activeButton.offsetLeft}px`;
    }
  }

  private initCustomModeSelector(): void {
    this.modeList.innerHTML = ''; // Clear existing
    for (const key in MODES) {
        const mode = MODES[key as ModeID];
        const optionButton = document.createElement('button');
        optionButton.className = 'mode-option';
        optionButton.textContent = mode.name;
        optionButton.dataset.modeId = mode.id;

        optionButton.addEventListener('click', () => {
            this.handleModeChange(mode.id);
            this.closeModeList();
        });

        optionButton.addEventListener('mouseenter', (e) => this.showModeTooltip(e, mode));
        optionButton.addEventListener('mouseleave', () => this.hideTooltip());

        this.modeList.appendChild(optionButton);
    }
  }

  private initTimezoneSelector(): void {
    const timezones = [
      'UTC', 'Europe/Warsaw', 'Europe/London', 'Europe/Paris', 'Europe/Berlin',
      'America/New_York', 'America/Chicago', 'America/Denver', 'America/Los_Angeles',
      'Asia/Tokyo', 'Asia/Dubai', 'Asia/Kolkata', 'Australia/Sydney'
    ];
    
    this.timezoneList.innerHTML = ''; // Clear existing
    timezones.forEach(tz => {
        const optionButton = document.createElement('button');
        optionButton.className = 'mode-option';
        optionButton.textContent = tz.replace(/_/g, ' ');
        optionButton.dataset.tz = tz;

        optionButton.addEventListener('click', () => {
            this.handleTimezoneChange(tz);
            this.closeTimezoneList();
        });

        optionButton.addEventListener('mouseenter', (e) => this.showTimezoneTooltip(e, tz));
        optionButton.addEventListener('mouseleave', () => this.hideTooltip());

        this.timezoneList.appendChild(optionButton);
    });
    
    const savedTimezone = localStorage.getItem('selectedTimezone');
    if (savedTimezone && timezones.includes(savedTimezone)) {
        this.currentTimezone = savedTimezone;
    } else {
        this.currentTimezone = 'Europe/Warsaw'; // Default
    }
    this.updateTimezoneDisplay();
  }

  private handleTimezoneChange(newTimezone: string): void {
    this.currentTimezone = newTimezone;
    localStorage.setItem('selectedTimezone', this.currentTimezone);
    this.updateTimezoneDisplay();
    this.updateMetadataDisplay();
  }
  
  private updateTimezoneDisplay(): void {
    if (this.currentTimezoneNameSpan) {
        this.currentTimezoneNameSpan.textContent = this.currentTimezone.replace(/_/g, ' ');
    }
    this.timezoneList.querySelectorAll('.mode-option').forEach(opt => {
        const button = opt as HTMLButtonElement;
        if (button.dataset.tz === this.currentTimezone) {
            button.classList.add('active');
        } else {
            button.classList.remove('active');
        }
    });
  }

  private toggleTimezoneList(): void {
    if (this.timezoneList.classList.contains('show')) {
        this.closeTimezoneList();
    } else {
        this.closeModeList(); // Close other dropdown for better UX
        this.timezoneList.classList.add('show');
        this.timezoneSelectorButton.classList.add('open');
    }
  }

  private closeTimezoneList(): void {
      this.timezoneList.classList.remove('show');
      this.timezoneSelectorButton.classList.remove('open');
  }

  private toggleModeList(): void {
    if (this.modeList.classList.contains('show')) {
        this.closeModeList();
    } else {
        this.closeTimezoneList(); // Close other dropdown for better UX
        this.closeMoreMenu();
        this.modeList.classList.add('show');
        this.modeSelectorButton.classList.add('open');
    }
  }

  private closeModeList(): void {
      this.modeList.classList.remove('show');
      this.modeSelectorButton.classList.remove('open');
  }

  private handleDocumentClick(event: MouseEvent): void {
      if (!this.modeSelectorContainer.contains(event.target as Node)) {
          this.closeModeList();
      }
      if (!this.timezoneSelectorContainer.contains(event.target as Node)) {
          this.closeTimezoneList();
      }
      if (!this.moreMenuContainer.contains(event.target as Node)) {
          this.closeMoreMenu();
      }
  }

  private showModeTooltip(event: MouseEvent, mode: Mode): void {
      const target = event.currentTarget as HTMLElement;
      const rect = target.getBoundingClientRect();
  
      this.modeTooltip.innerHTML = `<h4>${mode.name}</h4><pre>${mode.instructions}</pre>`;
      this.modeTooltip.classList.add('show');
      
      const tooltipRect = this.modeTooltip.getBoundingClientRect();
      const appContainer = document.querySelector('.app-container') as HTMLElement;
      const containerRect = appContainer.getBoundingClientRect();
  
      let left = rect.right + 10;
      if (left + tooltipRect.width > containerRect.right - 10) {
          left = rect.left - tooltipRect.width - 10;
      }
  
      this.modeTooltip.style.top = `${rect.top}px`;
      this.modeTooltip.style.left = `${left}px`;
  }
  
  private showTimezoneTooltip(event: MouseEvent, tz: string): void {
    const target = event.currentTarget as HTMLElement;
    const rect = target.getBoundingClientRect();

    const currentTime = new Date().toLocaleTimeString('en-US', {
        hour: '2-digit',
        minute: '2-digit',
        timeZoneName: 'short',
        timeZone: tz
    });

    this.modeTooltip.innerHTML = `<h4>${tz.replace(/_/g, ' ')}</h4><p>Current time: ${currentTime}</p>`;
    this.modeTooltip.classList.add('show');
    
    const tooltipRect = this.modeTooltip.getBoundingClientRect();
    const appContainer = document.querySelector('.app-container') as HTMLElement;
    const containerRect = appContainer.getBoundingClientRect();

    let left = rect.right + 10;
    if (left + tooltipRect.width > containerRect.right - 10) {
        left = rect.left - tooltipRect.width - 10;
    }

    this.modeTooltip.style.top = `${rect.top}px`;
    this.modeTooltip.style.left = `${left}px`;
  }

  private hideTooltip(): void {
      this.modeTooltip.classList.remove('show');
  }

  private loadAndSetInitialMode(): void {
    const savedMode = localStorage.getItem('selectedMode') as ModeID;
    if (savedMode && MODES[savedMode]) {
      this.currentModeId = savedMode;
    } else {
      this.currentModeId = 'journal'; // Default mode
    }
    this.updateModeDisplay();
    this.updateCustomPromptButtonVisibility();
  }
  
  private updateModeDisplay(): void {
    const currentMode = MODES[this.currentModeId];
    if (currentMode) {
        this.currentModeNameSpan.textContent = currentMode.name;
    }

    this.modeList.querySelectorAll('.mode-option').forEach(opt => {
        const button = opt as HTMLButtonElement;
        if (button.dataset.modeId === this.currentModeId) {
            button.classList.add('active');
        } else {
            button.classList.remove('active');
        }
    });
  }

  private handleModeChange(newModeId: ModeID): void {
    this.currentModeId = newModeId;
    localStorage.setItem('selectedMode', this.currentModeId);
    this.updateModeDisplay();
    if(this.currentNote) {
        this.currentNote.modeId = newModeId;
        this.updateMetadataDisplay();
    }
    this.updateCustomPromptButtonVisibility();
  }

  private loadCustomPrompt(): void {
    const savedPrompt = localStorage.getItem('customPromptInstructions');
    // A helpful default prompt for first-time users.
    this.customPromptInstructions = savedPrompt || `You are a helpful assistant. Please follow these instructions:
- Summarize the text into three bullet points.
- Identify any questions asked within the text.
- List all action items clearly using markdown checkboxes.`;
    MODES.custom.instructions = this.customPromptInstructions;
  }

  private updateCustomPromptButtonVisibility(): void {
    if (this.currentModeId === 'custom') {
        this.editCustomPromptButton.style.display = 'flex';
    } else {
        this.editCustomPromptButton.style.display = 'none';
    }
  }

  private openCustomPromptModal(): void {
    this.customPromptTextarea.value = this.customPromptInstructions;
    this.customPromptModal.style.display = 'flex';
    this.customPromptTextarea.focus();
  }

  private closeCustomPromptModal(): void {
    this.customPromptModal.style.display = 'none';
  }

  private saveCustomPrompt(): void {
    const newPrompt = this.customPromptTextarea.value.trim();
    if (newPrompt) {
      this.customPromptInstructions = newPrompt;
      MODES.custom.instructions = newPrompt;
      localStorage.setItem('customPromptInstructions', newPrompt);
      this.closeCustomPromptModal();
    } else {
      // Optional: show an error message that prompt can't be empty
      this.customPromptTextarea.placeholder = 'Prompt cannot be empty. Please enter your instructions.';
    }
  }

  private handleResize(): void {
    this.updateTabIndicator();
    if (this.isRecording && this.liveWaveformCanvas) {
      requestAnimationFrame(() => {
        this.setupCanvasDimensions();
      });
    }
  }

  private setupCanvasDimensions(): void {
    if (!this.liveWaveformCanvas || !this.liveWaveformCtx) return;

    const canvas = this.liveWaveformCanvas;
    const dpr = window.devicePixelRatio || 1;

    const rect = canvas.getBoundingClientRect();
    const cssWidth = rect.width;
    const cssHeight = rect.height;

    canvas.width = Math.round(cssWidth * dpr);
    canvas.height = Math.round(cssHeight * dpr);

    this.liveWaveformCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  private initTheme(): void {
    const savedTheme = localStorage.getItem('theme');
    const isLight = savedTheme === 'light';
    
    if (isLight) {
        document.body.classList.add('light-mode');
    } else {
        document.body.classList.remove('light-mode');
    }

    this.updateThemeIcons(isLight);
    this.updateThemeColorMeta();
  }

  private toggleTheme(): void {
    const isLight = document.body.classList.toggle('light-mode');
    localStorage.setItem('theme', isLight ? 'light' : 'dark');
    this.updateThemeIcons(isLight);
    this.updateThemeColorMeta();
  }
  
  private updateThemeIcons(isLight: boolean): void {
    const iconClassAdd = isLight ? 'fa-moon' : 'fa-sun';
    const iconClassRemove = isLight ? 'fa-sun' : 'fa-moon';

    this.themeToggleIcon.classList.remove(iconClassRemove);
    this.themeToggleIcon.classList.add(iconClassAdd);

    if (this.moreMenuThemeToggleIcon) {
        this.moreMenuThemeToggleIcon.classList.remove(iconClassRemove);
        this.moreMenuThemeToggleIcon.classList.add(iconClassAdd);
    }
  }

  private updateThemeColorMeta(): void {
    const themeColor = getComputedStyle(document.body).getPropertyValue('--color-surface').trim();
    const metaThemeColor = document.querySelector('meta[name="theme-color"]');
    if (metaThemeColor) {
      metaThemeColor.setAttribute('content', themeColor);
    }
  }

  private initMoreMenu(): void {
    this.moreMenuList.innerHTML = '';
    
    const actions = [
        { id: 'copy', icon: 'fa-copy', text: 'Copy Polished Note', action: () => this.copyPolishedNote() },
        { id: 'copyMeta', icon: 'fa-clipboard', text: 'Copy Metadata', action: () => this.copyMetadata() },
        { id: 'theme', icon: 'fa-sun', text: 'Toggle Theme', action: () => this.toggleTheme() }
    ];

    actions.forEach(action => {
        const button = document.createElement('button');
        button.className = 'more-menu-item';
        button.title = action.text;
        button.addEventListener('click', () => {
            action.action();
            this.closeMoreMenu();
        });

        const icon = document.createElement('i');
        icon.className = `fas ${action.icon}`;
        
        if (action.id === 'theme') {
            this.moreMenuThemeToggleIcon = icon;
        }

        const span = document.createElement('span');
        span.textContent = action.text;

        button.appendChild(icon);
        button.appendChild(span);
        this.moreMenuList.appendChild(button);
    });
  }

  private toggleMoreMenu(): void {
    this.closeModeList();
    this.closeTimezoneList();
    this.moreMenuList.classList.toggle('show');
  }

  private closeMoreMenu(): void {
      this.moreMenuList.classList.remove('show');
  }

  private async startFullRecordingSession(): Promise<void> {
    if (this.isRecording || this.isProcessing) return;

    this.isRecording = true;
    this.isPaused = false;
    this.lapCount = 0;
    this.allRawLapText = '';
    this.totalDurationMs = 0;
    
    if (this.currentNote) {
      this.currentNote.timestamp = Date.now();
      this.currentNote.duration = 0;
      this.currentNote.audioSize = 0;
      this.currentNote.promptTokens = 0;
      this.currentNote.completionTokens = 0;
      this.currentNote.cost = 0;
    }
    
    const rawPlaceholder = this.rawTranscription.getAttribute('placeholder') || '';
    this.rawTranscription.textContent = rawPlaceholder;
    this.rawTranscription.classList.add('placeholder-active');
    
    const polishedPlaceholder = this.polishedNote.getAttribute('placeholder') || '';
    this.polishedNote.innerHTML = polishedPlaceholder;
    this.polishedNote.classList.add('placeholder-active');

    this.updateMetadataDisplay();
    this.showRecordingDialog();
    await this._startNextRecordingSegment();
  }

  private async stopFullRecordingSession(): Promise<void> {
    if (!this.isRecording || this.isProcessing) return;
    this.stopReason = 'stop';
    this.mediaRecorder?.stop();
  }

  private async handleLap(): Promise<void> {
    if (!this.isRecording || this.isPaused || this.isProcessing) return;
    this.stopReason = 'lap';
    this.mediaRecorder?.stop();
  }

  private async handlePauseResume(): Promise<void> {
    if (!this.isRecording || this.isProcessing) return;

    const icon = this.pauseButton.querySelector('i');
    if (!icon) return;

    if (this.isPaused) { // RESUMING
      this.mediaRecorder?.resume();
      this.isPaused = false;
      icon.classList.remove('fa-play');
      icon.classList.add('fa-pause');
      this.liveRecordingTitle.textContent = 'Recording...';
      this.drawLiveWaveform(); // Restart waveform

      // Reset the start time for the current part of the segment
      this.recordingStartTime = Date.now();
      
      if (this.timerIntervalId) clearInterval(this.timerIntervalId);
      this.timerIntervalId = window.setInterval(() => this.updateLiveTimer(), 50);
    } else { // PAUSING
      this.mediaRecorder?.pause();
      this.isPaused = true;
      
      // Capture the elapsed time for this part of the segment and add it to the total
      const segmentPartDuration = Date.now() - this.recordingStartTime;
      this.totalDurationMs += segmentPartDuration;

      icon.classList.remove('fa-pause');
      icon.classList.add('fa-play');
      this.liveRecordingTitle.textContent = 'Paused';
      if (this.waveformDrawingId) cancelAnimationFrame(this.waveformDrawingId); // Stop waveform
      if (this.timerIntervalId) clearInterval(this.timerIntervalId); // Stop timer
    }
  }

  private setupAudioVisualizer(): void {
    if (!this.stream || this.audioContext) return;

    this.audioContext = new (window.AudioContext ||
      (window as any).webkitAudioContext)();
    const source = this.audioContext.createMediaStreamSource(this.stream);
    this.analyserNode = this.audioContext.createAnalyser();

    this.analyserNode.fftSize = 256;
    this.analyserNode.smoothingTimeConstant = 0.75;

    const bufferLength = this.analyserNode.frequencyBinCount;
    this.waveformDataArray = new Uint8Array(bufferLength);

    source.connect(this.analyserNode);
  }

  private drawLiveWaveform(): void {
    if (
      !this.analyserNode || !this.waveformDataArray || !this.liveWaveformCtx ||
      !this.liveWaveformCanvas || !this.isRecording || this.isPaused
    ) {
      if (this.waveformDrawingId) cancelAnimationFrame(this.waveformDrawingId);
      this.waveformDrawingId = null;
      return;
    }

    this.waveformDrawingId = requestAnimationFrame(() => this.drawLiveWaveform());
    this.analyserNode.getByteTimeDomainData(this.waveformDataArray);

    const ctx = this.liveWaveformCtx;
    const canvas = this.liveWaveformCanvas;
    const logicalWidth = canvas.clientWidth;
    const logicalHeight = canvas.clientHeight;
    ctx.clearRect(0, 0, logicalWidth, logicalHeight);
    
    const recordingColor = getComputedStyle(document.documentElement).getPropertyValue('--color-primary').trim();
    ctx.lineWidth = 2;
    ctx.strokeStyle = recordingColor;
    ctx.beginPath();

    const sliceWidth = logicalWidth * 1.0 / this.analyserNode.frequencyBinCount;
    let x = 0;

    for(let i = 0; i < this.analyserNode.frequencyBinCount; i++) {
        const v = this.waveformDataArray[i] / 128.0;
        const y = v * logicalHeight / 2;

        if(i === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }

        x += sliceWidth;
    }

    ctx.lineTo(canvas.clientWidth, canvas.clientHeight / 2);
    ctx.stroke();
  }

  private updateLiveTimer(): void {
    if (!this.isRecording || !this.liveRecordingTimerDisplay || this.isPaused) return;
    const now = Date.now();
    const elapsedMs = (now - this.recordingStartTime) + this.totalDurationMs;

    const totalSeconds = Math.floor(elapsedMs / 1000);
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = totalSeconds % 60;
    const hundredths = Math.floor((elapsedMs % 1000) / 10);

    this.liveRecordingTimerDisplay.textContent = `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}.${String(hundredths).padStart(2, '0')}`;
  }

  private showRecordingDialog(): void {
    this.fabRecord.style.display = 'none';
    this.bottomNav.style.display = 'none';
    this.recordingDialog.classList.add('show');
    
    this.setupCanvasDimensions();
    this.liveRecordingTitle.textContent = 'Recording...';
    
    this.recordingStartTime = Date.now();
    this.updateLiveTimer();
    if (this.timerIntervalId) clearInterval(this.timerIntervalId);
    this.timerIntervalId = window.setInterval(() => this.updateLiveTimer(), 50);
  }

  private hideRecordingDialog(): void {
    if (window.matchMedia("(max-width: 767px)").matches) {
      this.bottomNav.style.display = 'flex';
      this.fabRecord.style.display = 'none';
    } else {
      this.bottomNav.style.display = 'none';
      this.fabRecord.style.display = 'flex';
    }
    
    this.recordingDialog.classList.remove('show');
    
    if (this.waveformDrawingId) {
      cancelAnimationFrame(this.waveformDrawingId);
      this.waveformDrawingId = null;
    }
    if (this.timerIntervalId) {
      clearInterval(this.timerIntervalId);
      this.timerIntervalId = null;
    }
    if (this.liveWaveformCtx && this.liveWaveformCanvas) {
      this.liveWaveformCtx.clearRect(0, 0, this.liveWaveformCanvas.width, this.liveWaveformCanvas.height);
    }
    if (this.audioContext) {
      if (this.audioContext.state !== 'closed') {
        this.audioContext.close().catch((e) => console.warn('Error closing audio context', e));
      }
      this.audioContext = null;
    }
    this.analyserNode = null;
    this.waveformDataArray = null;
  }

  private async _startNextRecordingSegment(): Promise<void> {
    try {
      this.audioChunks = [];
      if (!this.stream) {
          this.recordingStatus.textContent = 'Requesting microphone...';
          try {
              this.stream = await navigator.mediaDevices.getUserMedia({audio: true});
          } catch (err) {
              console.error('Failed with basic constraints:', err);
              this.stream = await navigator.mediaDevices.getUserMedia({
                  audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
              });
          }
      }

      this.setupAudioVisualizer();
      this.drawLiveWaveform();
      this.mediaRecorder = new MediaRecorder(this.stream, { mimeType: 'audio/webm' });

      this.mediaRecorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0)
          this.audioChunks.push(event.data);
      };

      this.mediaRecorder.onstop = async () => {
        // If the recorder was paused, the pre-pause duration is already in totalDurationMs.
        // We only need to add the duration of the final active segment part.
        if (!this.isPaused) {
            const segmentDuration = Date.now() - this.recordingStartTime;
            this.totalDurationMs += segmentDuration;
        }
        
        if (this.audioChunks.length > 0) {
            const audioBlob = new Blob(this.audioChunks, { type: this.mediaRecorder?.mimeType || 'audio/webm' });
            if(this.currentNote) {
                this.currentNote.audioSize += audioBlob.size;
            }
            await this.processAudioSegment(audioBlob);
        } else { // No audio captured in this segment
            if (this.stopReason === 'lap') {
                await this._startNextRecordingSegment(); // Just restart for next lap
            } else { // Final stop with no final audio
                this.recordingStatus.textContent = 'Polishing note...';
                await this.getPolishedNote();
                this.resetToIdleState();
            }
        }
      };

      this.mediaRecorder.start();
      this.recordingStartTime = Date.now();
      this.liveRecordingTitle.textContent = 'Recording...';
      if (this.timerIntervalId) clearInterval(this.timerIntervalId);
      this.timerIntervalId = window.setInterval(() => this.updateLiveTimer(), 50);

    } catch (error) {
      console.error('Error starting recording:', error);
      this.recordingStatus.textContent = `Error: ${error instanceof Error ? error.message : "Unknown error"}`;
      this.resetToIdleState();
    }
  }

  private async processAudioSegment(audioBlob: Blob): Promise<void> {
    if (this.isProcessing) return;
    this.isProcessing = true;
    this.setLiveControls(false); // Disable buttons
    this.lapCount++;
    this.liveRecordingTitle.textContent = `Processing Lap ${this.lapCount}...`;
    if (this.timerIntervalId) clearInterval(this.timerIntervalId);

    try {
      const base64Audio = await this.blobToBase64(audioBlob);
      if (!base64Audio) throw new Error('Failed to convert audio');

      const mimeType = this.mediaRecorder?.mimeType || 'audio/webm';
      const transcriptionText = await this.getTranscription(base64Audio, mimeType);
      
      const segmentStartTime = this.formatDuration(this.totalDurationMs - (Date.now() - this.recordingStartTime));
      const segmentEndTime = this.formatDuration(this.totalDurationMs);
      const lapHeader = `\n\n--- LAP ${this.lapCount} (${segmentStartTime} - ${segmentEndTime}) ---\n\n`;
      this.allRawLapText += lapHeader + (transcriptionText || '[No speech detected]');
      
      this.rawTranscription.textContent = this.allRawLapText;
      if (this.rawTranscription.classList.contains('placeholder-active')) {
          this.rawTranscription.classList.remove('placeholder-active');
      }
      if(this.currentNote) this.currentNote.rawTranscription = this.allRawLapText;

    } catch (error) {
        console.error('Error processing audio segment:', error);
        this.recordingStatus.textContent = 'Error processing segment.';
    } finally {
        if (this.stopReason === 'lap') {
            await this._startNextRecordingSegment();
        } else { // 'stop'
            this.liveRecordingTitle.textContent = 'Polishing final note...';
            await this.getPolishedNote();
            this.resetToIdleState();
        }
        this.isProcessing = false;
        this.setLiveControls(true); // Re-enable buttons if continuing
    }
  }

  private triggerFileUpload(): void {
    if (this.isRecording || this.isProcessing) return;
    this.audioUploadInput.click();
  }

  private async handleFileUpload(event: Event): Promise<void> {
    const input = event.target as HTMLInputElement;
    if (!input.files || input.files.length === 0) {
        return;
    }
    const file = input.files[0];

    // Validation
    if (!file.type.startsWith('audio/')) {
        this.recordingStatus.textContent = 'Error: Invalid file type. Please upload an audio file.';
        input.value = ''; // Reset for next selection
        setTimeout(() => {
            if (this.recordingStatus.textContent?.includes('Invalid file type')) {
                this.recordingStatus.textContent = 'Ready to record';
            }
        }, 3000);
        return;
    }
    
    if (this.isRecording || this.isProcessing) {
        this.recordingStatus.textContent = 'Please wait for the current process to finish.';
        input.value = '';
        return;
    }
    
    this.createNewNote();
    
    this.isProcessing = true;
    this.fabRecord.disabled = true;

    try {
        if (this.currentNote) {
            this.currentNote.audioSize = file.size;
            this.currentNote.duration = 0; // Duration is not available for uploads
            this.totalDurationMs = 0;
        }
        this.updateMetadataDisplay();

        this.recordingStatus.textContent = `Processing ${file.name}...`;

        const base64Audio = await this.blobToBase64(file);
        if (!base64Audio) throw new Error('Failed to read the audio file.');

        const transcriptionText = await this.getTranscription(base64Audio, file.type, true);
        
        this.allRawLapText = transcriptionText || '[No speech detected]';
        this.rawTranscription.textContent = this.allRawLapText;
        if (this.rawTranscription.classList.contains('placeholder-active')) {
            this.rawTranscription.classList.remove('placeholder-active');
        }
        if(this.currentNote) this.currentNote.rawTranscription = this.allRawLapText;
        
        await this.getPolishedNote();
    } catch (error) {
        console.error('Error processing uploaded file:', error);
        const errorMessage = error instanceof Error ? error.message : "Upload failed";
        this.recordingStatus.textContent = `Error: ${errorMessage}`;
    } finally {
        this.isProcessing = false;
        this.fabRecord.disabled = false;
        input.value = ''; // Reset for next selection
        this.updateMetadataDisplay();
    }
  }

  private async blobToBase64(blob: Blob): Promise<string> {
      const reader = new FileReader();
      const readResult = new Promise<string>((resolve, reject) => {
        reader.onloadend = () => {
          try {
            resolve((reader.result as string).split(',')[1]);
          } catch (err) {
            reject(err);
          }
        };
        reader.onerror = () => reject(reader.error);
      });
      reader.readAsDataURL(blob);
      return readResult;
  }

  private async getTranscription(base64Audio: string, mimeType: string, isUpload: boolean = false): Promise<string> {
    const context = isUpload ? 'file' : `Lap ${this.lapCount}`;
    try {
      this.recordingStatus.textContent = `Transcribing ${context}...`;
      const contents = {
          parts: [
            {text: 'Transcribe this audio with the following format:\n\n[TIMESTAMP] SPEAKER: exact spoken words\n\nInclude timestamps every 10-15 seconds, detect different speakers (Speaker 1, Speaker 2, etc.), mark pauses with [PAUSE], unclear words with [UNCLEAR], and background sounds with [BACKGROUND: description]. Capture everything exactly as spoken including filler words, repetitions, and false starts.'},
            {inlineData: {mimeType: mimeType, data: base64Audio}},
          ],
      };
      const response = await this.genAI.models.generateContent({ model: MODEL_NAME, contents: contents });
      
      if (response.usageMetadata && this.currentNote) {
        this.currentNote.promptTokens += response.usageMetadata.promptTokenCount ?? 0;
        this.currentNote.completionTokens += response.usageMetadata.candidatesTokenCount ?? 0;
        this.updateNoteCost();
        this.updateMetadataDisplay();
      }
      return response.text;
    } catch (error) {
      console.error(`Error getting transcription for ${context}:`, error);
      this.recordingStatus.textContent = `Error transcribing ${context}.`;
      return `[Error during transcription: ${error instanceof Error ? error.message : String(error)}]`;
    }
  }

  private async getPolishedNote(): Promise<void> {
    try {
      if (!this.allRawLapText.trim()) {
        this.recordingStatus.textContent = 'No transcription to polish';
        this.polishedNote.innerHTML = '<p><em>No transcription available to polish.</em></p>';
        this.polishedNote.classList.add('placeholder-active');
        return;
      }
      this.recordingStatus.textContent = 'Polishing note...';
      const mode = MODES[this.currentModeId] || MODES.journal;
      const selectedTimezone = this.currentTimezone;
      const location = selectedTimezone.split('/').pop()?.replace(/_/g, ' ') || 'Unknown Location';
      const noteTimestamp = this.currentNote ? this.currentNote.timestamp : Date.now();
      const timestamp = new Date(noteTimestamp).toLocaleString('en-US', {
          timeZone: selectedTimezone, dateStyle: 'full', timeStyle: 'short',
      });
      const prompt = `You are a specialized AI assistant that transforms raw audio transcription into a specific, structured format based on the user's selected 'mode'.

Your task is to follow the instructions for the selected mode precisely and generate a markdown response.
The note MUST begin with the provided location and timestamp.
Do not add any commentary before or after the markdown content.

Location: ${location}
Timestamp: ${timestamp}
Mode: ${mode.name}
Instructions:
${mode.instructions}

---

Raw transcription (from multiple laps):
${this.allRawLapText}`;
      
      const response = await this.genAI.models.generateContent({ model: MODEL_NAME, contents: prompt });

      if (response.usageMetadata && this.currentNote) {
        this.currentNote.promptTokens += response.usageMetadata.promptTokenCount ?? 0;
        this.currentNote.completionTokens += response.usageMetadata.candidatesTokenCount ?? 0;
        this.updateNoteCost();
      }
      const polishedText = response.text;
      if (polishedText) {
        const htmlContent = await marked.parse(String(polishedText));
        this.polishedNote.innerHTML = htmlContent;
        this.polishedNote.classList.remove('placeholder-active');
        if (this.currentNote) this.currentNote.polishedNote = polishedText;
        this.recordingStatus.textContent = 'Note polished. Ready for next recording.';
      } else {
        this.recordingStatus.textContent = 'Polishing failed or returned empty.';
        this.polishedNote.innerHTML = '<p><em>Polishing returned empty. Raw transcription is available.</em></p>';
        this.polishedNote.classList.add('placeholder-active');
      }
    } catch (error) {
      console.error('Error polishing note:', error);
      this.recordingStatus.textContent = 'Error polishing note. Please try again.';
      this.polishedNote.innerHTML = `<p><em>Error during polishing: ${error instanceof Error ? error.message : String(error)}</em></p>`;
      this.polishedNote.classList.add('placeholder-active');
    } finally {
        this.updateMetadataDisplay();
    }
  }

  private setButtonState(button: HTMLButtonElement, state: 'success' | 'error'): void {
    const icon = button.querySelector('i');
    if (!icon) return;

    const originalIconClasses = button.dataset.originalIcon || icon.className;
    if (!button.dataset.originalIcon) {
        button.dataset.originalIcon = originalIconClasses;
    }

    button.classList.remove('copied', 'error'); // remove previous states
    button.classList.add(state === 'success' ? 'copied' : 'error');
    icon.className = `fas ${state === 'success' ? 'fa-check' : 'fa-times'}`;

    const existingTimeoutId = parseInt(button.dataset.timeoutId || '0', 10);
    if (existingTimeoutId) {
        clearTimeout(existingTimeoutId);
    }

    const timeoutId = window.setTimeout(() => {
        button.classList.remove('copied', 'error');
        icon.className = originalIconClasses;
        delete button.dataset.timeoutId;
        delete button.dataset.originalIcon;
    }, 2000);
    button.dataset.timeoutId = String(timeoutId);
  }

  private async copyPolishedNote(): Promise<void> {
    if (this.polishedNote.classList.contains('placeholder-active') || this.polishedNote.innerText.trim() === '') {
      console.warn('No polished note content to copy.');
      return;
    }
    
    try {
      const htmlBlob = new Blob([this.polishedNote.innerHTML], { type: 'text/html' });
      const textBlob = new Blob([this.polishedNote.innerText], { type: 'text/plain' });
      const item = new ClipboardItem({ 'text/html': htmlBlob, 'text/plain': textBlob });
      await navigator.clipboard.write([item]);
      this.setButtonState(this.copyButton, 'success');
    } catch (err) {
      console.error('Failed to copy rich text, falling back to plain text: ', err);
      try {
        await navigator.clipboard.writeText(this.polishedNote.innerText);
        this.setButtonState(this.copyButton, 'success');
      } catch (fallbackErr) {
        console.error('Failed to copy as plain text: ', fallbackErr);
        this.setButtonState(this.copyButton, 'error');
      }
    }
  }

  private async copyRawTranscription(): Promise<void> {
    const rawText = this.rawTranscription.textContent?.trim() || '';
    if (this.rawTranscription.classList.contains('placeholder-active') || rawText === '') {
        console.warn('No raw transcription content to copy.');
        return;
    }
    
    try {
        await navigator.clipboard.writeText(rawText);
        this.setButtonState(this.copyRawButton, 'success');
    } catch (err) {
        console.error('Failed to copy raw transcription: ', err);
        this.setButtonState(this.copyRawButton, 'error');
    }
  }

  private async copyMetadata(): Promise<void> {
      if (!this.currentNote || this.currentNote.duration === 0 && this.totalDurationMs === 0) {
          console.warn('No metadata to copy.');
          return;
      }
      const { timestamp, audioSize, modeId, promptTokens, completionTokens, cost } = this.currentNote;
      
      const metaString = [
          `Date & Time: ${new Date(timestamp).toLocaleString(undefined, { year: 'numeric', month: 'long', day: 'numeric', hour: 'numeric', minute: '2-digit', timeZone: this.currentTimezone})}`,
          `Recording Duration: ${this.formatDuration(this.totalDurationMs || this.currentNote.duration)}`,
          `Audio File Size: ${this.formatBytes(audioSize)}`,
          `Processing Mode: ${MODES[modeId].name}`,
          `Tokens (Prompt / Completion): ${promptTokens} / ${completionTokens}`,
          `Estimated Cost (USD): $${cost.toFixed(5)}`
      ].join('\n');

      try {
          await navigator.clipboard.writeText(metaString);
          this.setButtonState(this.copyMetaButton, 'success');
      } catch (err) {
          console.error('Failed to copy metadata: ', err);
          this.setButtonState(this.copyMetaButton, 'error');
      }
  }

  private formatDuration(ms: number): string {
    if (ms <= 0) return '00:00';
    const totalSeconds = Math.floor(ms / 1000);
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = totalSeconds % 60;
    return `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
  }

  private formatBytes(bytes: number, decimals = 2): string {
    if (bytes <= 0) return '--';
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
  }

  private updateNoteCost(): void {
    if (!this.currentNote) return;
    const { promptTokens, completionTokens } = this.currentNote;
    const promptCost = (promptTokens / 1000) * COST_PER_1K_PROMPT_TOKENS;
    const completionCost = (completionTokens / 1000) * COST_PER_1K_COMPLETION_TOKENS;
    this.currentNote.cost = promptCost + completionCost;
  }
  
  private resetMetadataDisplay(): void {
    this.metaDatetime.querySelector('span')!.textContent = '--';
    this.metaDuration.querySelector('span')!.textContent = '--';
    this.metaSize.querySelector('span')!.textContent = '--';
    this.metaMode.querySelector('span')!.textContent = '--';
    this.metaTokens.querySelector('span')!.textContent = '-- / --';
    this.metaCost.querySelector('span')!.textContent = '$0.00000';
  }

  private updateMetadataDisplay(isLive: boolean = false): void {
    if (!this.currentNote) {
        this.resetMetadataDisplay();
        return;
    };
    const { timestamp, audioSize, modeId, promptTokens, completionTokens, cost } = this.currentNote;
    const dtSpan = this.metaDatetime.querySelector('span')!;
    dtSpan.textContent = new Date(timestamp).toLocaleString(undefined, {
        year: 'numeric', month: 'long', day: 'numeric',
        hour: 'numeric', minute: '2-digit',
        timeZone: this.currentTimezone,
    });
    const durSpan = this.metaDuration.querySelector('span')!;
    const duration = this.isRecording ? this.totalDurationMs : (this.currentNote.duration || this.totalDurationMs);
    durSpan.textContent = isLive ? 'Recording...' : this.formatDuration(duration);
    const sizeSpan = this.metaSize.querySelector('span')!;
    sizeSpan.textContent = isLive ? '...' : this.formatBytes(audioSize);
    const modeSpan = this.metaMode.querySelector('span')!;
    modeSpan.textContent = MODES[modeId].name;
    const tokensSpan = this.metaTokens.querySelector('span')!;
    tokensSpan.textContent = (promptTokens > 0 || completionTokens > 0) ? `${promptTokens} / ${completionTokens}` : '-- / --';
    const costSpan = this.metaCost.querySelector('span')!;
    costSpan.textContent = (cost > 0) ? `$${cost.toFixed(5)}` : '$0.00000';
    if(this.currentNote) this.currentNote.duration = this.totalDurationMs;
  }

  private resetToIdleState(): void {
    this.isRecording = false;
    this.isPaused = false;
    this.isProcessing = false;
    this.hideRecordingDialog();
    
    if (this.stream) {
        this.stream.getTracks().forEach(track => track.stop());
        this.stream = null;
    }
    this.mediaRecorder = null;
    this.updateMetadataDisplay();
  }

  private setLiveControls(enabled: boolean): void {
    this.stopButton.disabled = !enabled;
    this.pauseButton.disabled = !enabled;
    this.lapButton.disabled = !enabled;
  }

  private createNewNote(): void {
    if(this.isRecording) {
        this.stopFullRecordingSession();
    }

    this.currentNote = {
      id: `note_${Date.now()}`,
      rawTranscription: '',
      polishedNote: '',
      timestamp: Date.now(),
      duration: 0,
      audioSize: 0,
      modeId: this.currentModeId,
      promptTokens: 0,
      completionTokens: 0,
      cost: 0,
    };
    
    this.allRawLapText = '';
    this.totalDurationMs = 0;

    const rawPlaceholder = this.rawTranscription.getAttribute('placeholder') || '';
    this.rawTranscription.textContent = rawPlaceholder;
    this.rawTranscription.classList.add('placeholder-active');
    const polishedPlaceholder = this.polishedNote.getAttribute('placeholder') || '';
    this.polishedNote.innerHTML = polishedPlaceholder;
    this.polishedNote.classList.add('placeholder-active');

    this.resetMetadataDisplay();
    this.recordingStatus.textContent = 'Ready to record';
    this.resetToIdleState();
  }
}

document.addEventListener('DOMContentLoaded', () => {
  new VoiceNotesApp();
  document.querySelectorAll<HTMLElement>('[contenteditable][placeholder]').forEach((el) => {
      const placeholder = el.getAttribute('placeholder')!;
      function updatePlaceholderState() {
        const currentText = (el.id === 'polishedNote' ? el.innerText : el.textContent)?.trim();
        if (currentText === '' || currentText === placeholder) {
          if (el.id === 'polishedNote' && currentText === '') el.innerHTML = placeholder;
          else if (currentText === '') el.textContent = placeholder;
          el.classList.add('placeholder-active');
        } else {
          el.classList.remove('placeholder-active');
        }
      }
      updatePlaceholderState();
      el.addEventListener('focus', function () {
        const currentText = (this.id === 'polishedNote' ? this.innerText : this.textContent)?.trim();
        if (currentText === placeholder) {
          if (this.id === 'polishedNote') this.innerHTML = '';
          else this.textContent = '';
          this.classList.remove('placeholder-active');
        }
      });
      el.addEventListener('blur', () => updatePlaceholderState());
    });
});

export {};