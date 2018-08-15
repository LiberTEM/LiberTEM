import { ConfigState } from "./reducers";

const CONFIG_KEY = "LiberTEM.config";

export function joinPaths(config: ConfigState, ...parts: string[]) {
    return parts.join(config.separator);
}

export function mergeLocalStorage(config: ConfigState): ConfigState {
    let localSettings = window.localStorage.getItem(CONFIG_KEY);
    if (localSettings === null) {
        return config;
    }
    localSettings = JSON.parse(localSettings);
    const mergedConfig = Object.assign({}, config, localSettings);
    return mergedConfig;
}

export function setLocalStorage(config: ConfigState): void {
    const localSettings = ["cwd"].reduce((acc, item: keyof ConfigState) => {
        acc[item] = config[item];
        return acc;
    }, {} as ConfigState);
    window.localStorage.setItem(CONFIG_KEY, JSON.stringify(localSettings));
}

export function clearLocalStorage(): void {
    window.localStorage.removeItem(CONFIG_KEY);
}