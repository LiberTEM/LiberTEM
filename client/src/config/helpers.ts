import { MsgPartConfig } from "../messages";
import { ConfigState } from "./reducers";

const CONFIG_KEY = "LiberTEM.config";

export function joinPaths(config: ConfigState, ...parts: string[]) {
    return parts.join(config.separator);
}

export function mergeLocalStorage(config: MsgPartConfig): ConfigState {
    const localSettings = window.localStorage.getItem(CONFIG_KEY);
    if (localSettings === null) {
        return getDefaultLocalConfig(config);
    }
    const localSettingsParsed = JSON.parse(localSettings);
    const defaultConfig = getDefaultLocalConfig(config);
    const mergedConfig: ConfigState = Object.assign(defaultConfig, localSettingsParsed);
    return mergedConfig;
}

export function setLocalStorage(config: ConfigState): void {
    const localSettings = ["cwd", "lastOpened"].reduce((acc, item: keyof ConfigState) => {
        acc[item] = config[item];
        return acc;
    }, {} as ConfigState);
    window.localStorage.setItem(CONFIG_KEY, JSON.stringify(localSettings));
}

export function clearLocalStorage(): void {
    window.localStorage.removeItem(CONFIG_KEY);
}

export function getDefaultLocalConfig(config: MsgPartConfig): ConfigState {
    return Object.assign({}, config, {
        lastOpened: {},
        cwd: "/",
    });
}