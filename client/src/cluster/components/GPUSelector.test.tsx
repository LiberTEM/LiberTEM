import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { ConfigState } from '../../config/reducers';
import { ClusterTypes } from '../../messages';
import { GPUSelector } from './GPUSelector';

describe('GPUSelector', () => {
    it('renders for zero cuda devices', () => {
        const config: ConfigState = {
            version: "v0.1.2",
            revision: "abcdef",
            localCores: 7,
            separator: '/',
            cwd: '/data/',
            datasetTypes: {
                'MIB': {
                    schema: {},
                    default_io_backend: null,
                    supported_io_backends: [],
                },
            },
            devices: {
                cpus: [0, 1, 2, 3, 4, 5, 6],
                cudas: [],
                has_cupy: false,
            },
            resultFileFormats: {
                NPZ: {
                    identifier: 'NPZ',
                    description: 'numpy format (.npz)',
                },
            },
            fileHistory: [],
            lastOpened: {},
            lastConnection: {
                type: ClusterTypes.LOCAL,
                address: 'tcp://localhost:8786',
                cudas: {},
            },
            starred: [],
            haveConfig: true,
        };
        const value = {
        };
        const setFieldValue = (name: string, value: any) => { };
        render(
            <GPUSelector config={config} name="gpus" value={value} setFieldValue={setFieldValue} />
        );
        expect(screen.queryByText(/Number of workers for CUDA device/)).toBeNull();
        expect(screen.queryByRole('spinbutton')).toBeNull();
    });

    it('renders warning if no cupy is found', () => {
        const config: ConfigState = {
            version: "v0.1.2",
            revision: "abcdef",
            localCores: 7,
            separator: '/',
            cwd: '/data/',
            datasetTypes: {
                'MIB': {
                    schema: {},
                    default_io_backend: null,
                    supported_io_backends: [],
                },
            },
            devices: {
                cpus: [0, 1, 2, 3, 4, 5, 6],
                cudas: [0, 1, 2],   // we have cuda devices, ...
                has_cupy: false,    // but no working cupy installation
            },
            resultFileFormats: {
                NPZ: {
                    identifier: 'NPZ',
                    description: 'numpy format (.npz)',
                },
            },
            fileHistory: [],
            lastOpened: {},
            lastConnection: {
                type: ClusterTypes.LOCAL,
                address: 'tcp://localhost:8786',
                cudas: {0: 42},
            },
            starred: [],
            haveConfig: true,
        };
        const value = {
            0: 42,
            1: 1,
            2: 1,
        };
        const setFieldValue = (name: string, value: any) => { };
        render(
            <GPUSelector config={config} name="gpus" value={value} setFieldValue={setFieldValue} />
        );

        expect(screen.getByText(/No cupy installation found/)).toBeInTheDocument();
        expect(screen.getAllByText(/Number of workers for CUDA device/).length).toBe(3);
        screen.getAllByRole('spinbutton').forEach(btn => {
            expect(btn).toBeDisabled();
        });
    });

    it('renders for one cuda device', () => {
        const config: ConfigState = {
            version: "v0.1.2",
            revision: "abcdef",
            localCores: 7,
            separator: '/',
            cwd: '/data/',
            datasetTypes: {
                'MIB': {
                    schema: {},
                    default_io_backend: null,
                    supported_io_backends: [],
                },
            },
            devices: {
                cpus: [0, 1, 2, 3, 4, 5, 6],
                cudas: [0],
                has_cupy: true,
            },
            resultFileFormats: {
                NPZ: {
                    identifier: 'NPZ',
                    description: 'numpy format (.npz)',
                },
            },
            fileHistory: [],
            lastOpened: {},
            lastConnection: {
                type: ClusterTypes.LOCAL,
                address: 'tcp://localhost:8786',
                cudas: {0: 42},
            },
            starred: [],
            haveConfig: true,
        };
        const value = {
            0: 42,  // initial value from lastConnection
        };
        const setFieldValue = (name: string, value: any) => { };
        render(
            <GPUSelector config={config} name="gpus" value={value} setFieldValue={setFieldValue} />
        );

        expect(screen.getByText(/Number of workers for CUDA device 0:/)).toBeInTheDocument();
        expect(screen.getByRole('spinbutton')).toBeInTheDocument();
    });

    it('renders for more cuda devices', () => {
        const config: ConfigState = {
            version: "v0.1.2",
            revision: "abcdef",
            localCores: 7,
            separator: '/',
            cwd: '/data/',
            datasetTypes: {
                'MIB': {
                    schema: {},
                    default_io_backend: null,
                    supported_io_backends: [],
                },
            },
            devices: {
                cpus: [0, 1, 2, 3, 4, 5, 6],
                cudas: [0, 1],
                has_cupy: true,
            },
            resultFileFormats: {
                NPZ: {
                    identifier: 'NPZ',
                    description: 'numpy format (.npz)',
                },
            },
            fileHistory: [],
            lastOpened: {},
            lastConnection: {
                type: ClusterTypes.LOCAL,
                address: 'tcp://localhost:8786',
                cudas: {0: 42},
            },
            starred: [],
            haveConfig: true,
        };
        const value = {
            0: 42,  // initial value from lastConnection
            1: 1,
        };
        const setFieldValue = (name: string, value: any) => { };
        render(
            <GPUSelector config={config} name="gpus" value={value} setFieldValue={setFieldValue} />
        );

        expect(screen.getByText(/Number of workers for CUDA device 0:/)).toBeInTheDocument();
        expect(screen.getByText(/Number of workers for CUDA device 1:/)).toBeInTheDocument();
        expect(screen.getAllByRole('spinbutton').length).toBe(2);
    });
})