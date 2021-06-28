import * as React from "react";
import { Icon } from "semantic-ui-react";

const BusySpinner: React.FC = () => {
    const styles: React.CSSProperties = {
        margin: 0,
        padding: 0,
        position: "absolute",
        bottom: "10px",
        right: "10px",
        color: "white",
        opacity: 0.7,
        filter: "drop-shadow(0 0 3px #000)",
    };
    return <Icon name="cog" loading style={styles} />
}

export default BusySpinner;