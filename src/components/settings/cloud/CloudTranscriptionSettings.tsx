import React, { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import {
  commands,
  type CloudApiMode,
  type CloudTranscriptionSettings as CloudSettings,
} from "@/bindings";
import { Alert } from "../../ui/Alert";
import { Button } from "../../ui/Button";
import { Dropdown } from "../../ui/Dropdown";
import { Input } from "../../ui/Input";
import { SettingContainer } from "../../ui/SettingContainer";
import { SettingsGroup } from "../../ui/SettingsGroup";
import { Textarea } from "../../ui/Textarea";
import { ToggleSwitch } from "../../ui/ToggleSwitch";
import { useSettings } from "../../../hooks/useSettings";
import { useSettingsStore } from "../../../stores/settingsStore";

/** One-click starting points; every field stays editable afterwards. */
const PRESETS: {
  id: string;
  label: string;
  api_mode: CloudApiMode;
  base_url: string;
  model: string;
}[] = [
  {
    id: "openai_audio",
    label: "OpenAI · gpt-4o-mini-audio-preview",
    api_mode: "chat_audio",
    base_url: "https://api.openai.com/v1",
    model: "gpt-4o-mini-audio-preview",
  },
  {
    id: "gemini",
    label: "Google Gemini · gemini-2.5-flash",
    api_mode: "chat_audio",
    base_url: "https://generativelanguage.googleapis.com/v1beta/openai",
    model: "gemini-2.5-flash",
  },
  {
    id: "openrouter",
    label: "OpenRouter · google/gemini-2.5-flash",
    api_mode: "chat_audio",
    base_url: "https://openrouter.ai/api/v1",
    model: "google/gemini-2.5-flash",
  },
  {
    id: "openai_transcribe",
    label: "OpenAI · gpt-4o-transcribe",
    api_mode: "transcriptions",
    base_url: "https://api.openai.com/v1",
    model: "gpt-4o-transcribe",
  },
  {
    id: "groq",
    label: "Groq · whisper-large-v3-turbo",
    api_mode: "transcriptions",
    base_url: "https://api.groq.com/openai/v1",
    model: "whisper-large-v3-turbo",
  },
];

/** A text field that saves when it loses focus. */
const BlurField: React.FC<{
  value: string;
  onCommit: (value: string) => void;
  type?: "text" | "password";
  placeholder?: string;
  disabled?: boolean;
}> = ({ value, onCommit, type = "text", placeholder, disabled }) => {
  const [local, setLocal] = useState(value);
  useEffect(() => setLocal(value), [value]);
  return (
    <Input
      type={type}
      value={local}
      onChange={(e) => setLocal(e.target.value)}
      onBlur={() => local !== value && onCommit(local)}
      placeholder={placeholder}
      variant="compact"
      disabled={disabled}
      className="flex-1 min-w-[320px]"
    />
  );
};

export const CloudTranscriptionSettings: React.FC = () => {
  const { t } = useTranslation();
  const { getSetting, updateSetting, isUpdating } = useSettings();
  const defaults = useSettingsStore(
    (s) => s.defaultSettings?.cloud_transcription,
  );
  const cloud: CloudSettings = getSetting("cloud_transcription") ?? {};
  const postProcessEnabled = getSetting("post_process_enabled") ?? false;
  const prompts = getSetting("post_process_prompts") ?? [];
  const selectedPromptId = getSetting("post_process_selected_prompt_id");
  const selectedPrompt = prompts.find((p) => p.id === selectedPromptId);

  const [promptDraft, setPromptDraft] = useState(cloud.prompt ?? "");
  useEffect(() => setPromptDraft(cloud.prompt ?? ""), [cloud.prompt]);

  const [testState, setTestState] = useState<
    | { kind: "idle" }
    | { kind: "running" }
    | { kind: "ok"; text: string }
    | { kind: "error"; message: string }
  >({ kind: "idle" });

  const save = (patch: Partial<CloudSettings>) =>
    updateSetting("cloud_transcription", { ...cloud, ...patch });

  const updating = isUpdating("cloud_transcription");
  const mode: CloudApiMode = cloud.api_mode ?? "chat_audio";
  const enabled = cloud.enabled ?? false;
  const defaultPrompt = defaults?.prompt ?? "";

  const runTest = async () => {
    setTestState({ kind: "running" });
    try {
      const result = await commands.testCloudTranscription();
      if (result.status === "ok") {
        setTestState({ kind: "ok", text: result.data });
      } else {
        setTestState({ kind: "error", message: result.error });
      }
    } catch (e) {
      setTestState({ kind: "error", message: String(e) });
    }
  };

  return (
    <div className="max-w-3xl w-full mx-auto space-y-6">
      <SettingsGroup title={t("settings.cloud.title")}>
        <ToggleSwitch
          checked={enabled}
          onChange={(value) => save({ enabled: value })}
          isUpdating={updating}
          label={t("settings.cloud.enabled.label")}
          description={t("settings.cloud.enabled.description")}
          descriptionMode="inline"
          grouped={true}
        />

        <SettingContainer
          title={t("settings.cloud.preset.title")}
          description={t("settings.cloud.preset.description")}
          descriptionMode="tooltip"
          layout="horizontal"
          grouped={true}
        >
          <Dropdown
            options={PRESETS.map((p) => ({ value: p.id, label: p.label }))}
            selectedValue={
              PRESETS.find(
                (p) =>
                  p.base_url === cloud.base_url &&
                  p.model === cloud.model &&
                  p.api_mode === mode,
              )?.id ?? null
            }
            placeholder={t("settings.cloud.preset.placeholder")}
            onSelect={(id) => {
              const preset = PRESETS.find((p) => p.id === id);
              if (preset)
                save({
                  api_mode: preset.api_mode,
                  base_url: preset.base_url,
                  model: preset.model,
                });
            }}
            disabled={updating}
            className="min-w-[320px]"
          />
        </SettingContainer>

        <SettingContainer
          title={t("settings.cloud.apiMode.title")}
          description={t("settings.cloud.apiMode.description")}
          descriptionMode="tooltip"
          layout="horizontal"
          grouped={true}
        >
          <Dropdown
            options={[
              {
                value: "chat_audio",
                label: t("settings.cloud.apiMode.chatAudio"),
              },
              {
                value: "transcriptions",
                label: t("settings.cloud.apiMode.transcriptions"),
              },
            ]}
            selectedValue={mode}
            onSelect={(value) => save({ api_mode: value as CloudApiMode })}
            disabled={updating}
            className="min-w-[320px]"
          />
        </SettingContainer>

        <SettingContainer
          title={t("settings.cloud.baseUrl.title")}
          description={t("settings.cloud.baseUrl.description")}
          descriptionMode="tooltip"
          layout="horizontal"
          grouped={true}
        >
          <BlurField
            value={cloud.base_url ?? ""}
            onCommit={(value) => save({ base_url: value.trim() })}
            placeholder="https://api.openai.com/v1"
            disabled={updating}
          />
        </SettingContainer>

        <SettingContainer
          title={t("settings.cloud.apiKey.title")}
          description={t("settings.cloud.apiKey.description")}
          descriptionMode="tooltip"
          layout="horizontal"
          grouped={true}
        >
          <BlurField
            type="password"
            value={cloud.api_key ?? ""}
            onCommit={(value) => save({ api_key: value.trim() })}
            placeholder={t("settings.cloud.apiKey.placeholder")}
            disabled={updating}
          />
        </SettingContainer>

        <SettingContainer
          title={t("settings.cloud.model.title")}
          description={t("settings.cloud.model.description")}
          descriptionMode="tooltip"
          layout="horizontal"
          grouped={true}
        >
          <BlurField
            value={cloud.model ?? ""}
            onCommit={(value) => save({ model: value.trim() })}
            placeholder="gpt-4o-mini-audio-preview"
            disabled={updating}
          />
        </SettingContainer>

        <SettingContainer
          title={t("settings.cloud.test.title")}
          description={t("settings.cloud.test.description")}
          descriptionMode="tooltip"
          layout="horizontal"
          grouped={true}
        >
          <Button
            variant="secondary"
            size="md"
            onClick={runTest}
            disabled={testState.kind === "running"}
          >
            {testState.kind === "running"
              ? t("settings.cloud.test.running")
              : t("settings.cloud.test.button")}
          </Button>
        </SettingContainer>
        {testState.kind === "ok" && (
          <Alert variant="success" contained>
            {testState.text.trim()
              ? t("settings.cloud.test.okWithText", { text: testState.text })
              : t("settings.cloud.test.ok")}
          </Alert>
        )}
        {testState.kind === "error" && (
          <Alert variant="error" contained>
            {testState.message}
          </Alert>
        )}
      </SettingsGroup>

      <SettingsGroup title={t("settings.cloud.prompt.groupTitle")}>
        <SettingContainer
          title={
            mode === "chat_audio"
              ? t("settings.cloud.prompt.titleChat")
              : t("settings.cloud.prompt.titleTranscriptions")
          }
          description={
            mode === "chat_audio"
              ? t("settings.cloud.prompt.descriptionChat")
              : t("settings.cloud.prompt.descriptionTranscriptions")
          }
          descriptionMode="inline"
          layout="stacked"
          grouped={true}
        >
          <div className="space-y-2">
            <Textarea
              value={promptDraft}
              onChange={(e) => setPromptDraft(e.target.value)}
              onBlur={() =>
                promptDraft !== (cloud.prompt ?? "") &&
                save({ prompt: promptDraft })
              }
            />
            <div className="flex gap-2">
              <Button
                variant="secondary"
                size="sm"
                disabled={updating || promptDraft === defaultPrompt}
                onClick={() => {
                  setPromptDraft(defaultPrompt);
                  save({ prompt: defaultPrompt });
                }}
              >
                {t("settings.cloud.prompt.reset")}
              </Button>
            </div>
          </div>
        </SettingContainer>

        {mode === "chat_audio" && (
          <ToggleSwitch
            checked={cloud.always_post_process ?? false}
            onChange={(value) => save({ always_post_process: value })}
            isUpdating={updating}
            label={t("settings.cloud.alwaysPostProcess.label")}
            description={t("settings.cloud.alwaysPostProcess.description")}
            descriptionMode="inline"
            grouped={true}
          />
        )}

        <Alert variant="info" contained>
          {mode === "chat_audio"
            ? selectedPrompt
              ? t("settings.cloud.postProcessInfo.chatWithPrompt", {
                  prompt: selectedPrompt.name,
                })
              : t("settings.cloud.postProcessInfo.chatNoPrompt")
            : t("settings.cloud.postProcessInfo.transcriptions")}
          {!postProcessEnabled &&
            " " + t("settings.cloud.postProcessInfo.enableHint")}
        </Alert>
      </SettingsGroup>
    </div>
  );
};
