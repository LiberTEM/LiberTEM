import * as React from "react";
import { Icon, SemanticICONS } from "semantic-ui-react";
import { DirectoryListingDetails } from "../../messages";


interface FileBrowserEntryProps {
    style: Record<string, unknown>,
    details: DirectoryListingDetails,
    onClick?: () => void,
    icon?: SemanticICONS,
}

interface TimeStampProps {
    ts: number,
}

export const TimeStamp: React.FC<TimeStampProps> = ({ ts }) => {
    const date = new Date(ts * 1000);
    const fmtDate = date.toLocaleDateString();
    const fmtTime = date.toLocaleTimeString();
    const title = `${fmtDate} ${fmtTime}`;
    return (
        <div style={{ display: "flex", whiteSpace: "nowrap" }} title={title}>
            <div style={{ width: "50%", marginRight: "10px" }}>{fmtDate}</div>
            <div style={{ width: "45%" }}>{fmtTime}</div>
        </div>
    )
}

// adapted from https://stackoverflow.com/a/14919494/540644
const humanFileSize = (bytes: number, si = false) => {
    const thresh = si ? 1000 : 1024;
    if (Math.abs(bytes) < thresh) {
        return {
            size: bytes,
            unit: 'B',
        }
    }
    const units = si
        ? ['kB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB']
        : ['KiB', 'MiB', 'GiB', 'TiB', 'PiB', 'EiB', 'ZiB', 'YiB'];
    let u = -1;
    do {
        bytes /= thresh;
        ++u;
    } while (Math.abs(bytes) >= thresh && u < units.length - 1);

    return {
        size: bytes.toFixed(1),
        unit: units[u],
    }
}

interface FileSizeProps {
    size: number,
    si?: boolean,
}

export const FileSize: React.FC<FileSizeProps> = ({ size, si }) => {
    const fmtSize = humanFileSize(size, si)
    return (
        <div style={{ textAlign: "right" }}>
            {fmtSize.size} {fmtSize.unit}
        </div>
    );
}

export const Cell: React.FC<{ title?: string, children?: React.ReactNode }> = ({ children, title }) => {
    const styles: React.CSSProperties = {
        whiteSpace: "nowrap",
        overflow: "hidden",
        textOverflow: "ellipsis",
        marginRight: "10px",
    }
    return (
        <div style={styles} title={title}>{children}</div>
    );
}


class FileBrowserEntry extends React.Component<FileBrowserEntryProps> {
    public onClick = (): void => {
        const { onClick } = this.props;
        if (onClick) {
            onClick();
        }
    }

    public render(): JSX.Element {
        const { details, style, icon } = this.props;
        const myStyle: React.CSSProperties = {
            cursor: "pointer",
            ...style,
        };

        const tsStyles: React.CSSProperties = {
            textAlign: "right",
        };

        return (
            <div onClick={this.onClick} style={myStyle}>
                <div style={{ display: "flex", paddingRight: "10px" }}>
                    <div style={{ width: "20%", flexGrow: 1 }}>
                        <Cell title={details.name}>
                            {icon && <Icon name={icon} />}
                            {details.name}
                        </Cell>
                    </div>
                    <div style={{ width: "10%" }}><Cell><FileSize size={details.size} si={false} /></Cell></div>
                    <div style={{ width: "10%" }}><Cell>{details.owner}</Cell></div>
                    <div style={{ width: "18%", ...tsStyles }}>
                        <Cell><TimeStamp ts={details.ctime} /></Cell>
                    </div>
                    <div style={{ width: "18%", ...tsStyles }}>
                        <Cell><TimeStamp ts={details.mtime} /></Cell>
                    </div>
                </div>
            </div>
        );
    }
}

export default FileBrowserEntry;