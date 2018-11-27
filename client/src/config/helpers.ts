import * as _ from 'lodash';
import { MsgPartConfig } from "../messages";
import { ConfigState } from "./reducers";

const CONFIG_KEY = "LiberTEM.config";

export function joinPaths(config: ConfigState, ...parts: string[]) {
    const removeSep = (part: string) => part.replace(new RegExp(`${_.escapeRegExp(config.separator)}$`), "")
    parts = [removeSep(parts[0]), ...parts.slice(1)];
    return parts.map(part => part.trim()).join(config.separator);
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
    const localSettings = ["cwd", "lastOpened", "fileHistory"].reduce((acc, item: keyof ConfigState) => {
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
        fileHistory: [],
        cwd: "/",
    });
}

export function makeUnique<T>(inp: T[]): T[] {
    return inp.reduce((acc: T[], curr: T) => {
        if (acc.indexOf(curr) === -1) {
            return [...acc, curr];
        } else {
            return acc;
        }
    }, [] as T[]);
}