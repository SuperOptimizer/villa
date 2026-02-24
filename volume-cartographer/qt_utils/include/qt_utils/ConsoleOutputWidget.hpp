#pragma once

#include <QWidget>

class QLabel;
class QPlainTextEdit;
class QPushButton;

namespace qt_utils
{

/// A read-only console output widget with clear and copy-to-clipboard buttons.
/// Displays monospaced text with auto-scrolling and a configurable title label.
class ConsoleOutputWidget : public QWidget
{
    Q_OBJECT

public:
    explicit ConsoleOutputWidget(QWidget* parent = nullptr);
    explicit ConsoleOutputWidget(const QString& title, QWidget* parent = nullptr);
    ~ConsoleOutputWidget() override;

    /// Append a line of text and auto-scroll to the bottom.
    void appendText(const QString& text);

    /// Clear all text.
    void clear();

    /// Set the title label text.
    void setTitle(const QString& title);

    /// Return the full contents as plain text.
    [[nodiscard]] auto text() const -> QString;

public slots:
    /// Copy all text to the system clipboard.
    void copyToClipboard();

signals:
    /// Emitted after clear() is called.
    void cleared();

private:
    void init(const QString& title);

    QPlainTextEdit* textEdit_{nullptr};
    QPushButton* clearButton_{nullptr};
    QPushButton* copyButton_{nullptr};
    QLabel* titleLabel_{nullptr};
};

}  // namespace qt_utils
