import * as React from "react";
import { Icon, SemanticICONS } from "semantic-ui-react";
import { DirectoryListingDetails } from "../../messages";


interface FileBrowserEntryProps {
    style: object,
    details: DirectoryListingDetails,
    onClick?: () => void,
    icon?: SemanticICONS,
}

interface TimeStampProps {
    ts: number,
}

export const TimeStamp: React.SFC<TimeStampProps> = ({ ts }) => {
    const date = new Date(ts * 1000);
    const fmtDate = date.toLocaleDateString();
    const fmtTime = date.toLocaleTimeString();
    return (
        <div style={{ display: "flex", whiteSpace: "nowrap" }}>
            <div style={{ width: "50%", marginRight: "10px" }}>{fmtDate}</div>
            <div style={{ width: "45%" }}>{fmtTime}</div>
        </div>
    )
}

class FileBrowserEntry extends React.Component<FileBrowserEntryProps> {
    public onClick = (e: React.MouseEvent) => {
        const { onClick } = this.props;
        if (onClick) {
            onClick();
        }
    }

    public render() {
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
                        {icon && <Icon name={icon} />}
                        {details.name}
                    </div>
                    <div style={{ width: "15%" }}>{details.owner}</div>
                    <div style={{ width: "18%", ...tsStyles }}><TimeStamp ts={details.ctime} /></div>
                    <div style={{ width: "18%", ...tsStyles }}><TimeStamp ts={details.mtime} /></div>
                </div>
            </div>
        );
    }
}

export default FileBrowserEntry;